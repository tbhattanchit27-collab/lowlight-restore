"""
Fine-tune the pretrained Zero-DCE on our training split.

Training data
  * synthetic pairs (data/synth/train)  -> zero-reference losses + L1 to ground truth
  * real dark photos (DICM train split) -> zero-reference losses only (no ground truth)
Validation
  * PSNR on data/synth/val after every epoch, best epoch is saved
Test data is never touched here.

The four zero-reference losses are the ones from the Zero-DCE paper:
  spatial consistency, exposure control (target 0.6), colour constancy,
  illumination smoothness (TV on the curve maps). Rewritten to run on CPU.

usage: python scripts/finetune_zerodce.py --epochs 30
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import read_rgb  # noqa: E402
from src.metrics import psnr  # noqa: E402
from src.zerodce import ZeroDCE, apply_curves  # noqa: E402

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# ------------------------------ losses --------------------------------------
def loss_color(x):
    m = x.mean(dim=(2, 3))
    r, g, b = m[:, 0], m[:, 1], m[:, 2]
    return torch.sqrt((r - g) ** 4 + (r - b) ** 4 + (g - b) ** 4 + 1e-8).mean()


def loss_exposure(x, patch=16, target=0.6):
    m = F.avg_pool2d(x.mean(1, keepdim=True), patch)
    return ((m - target) ** 2).mean()


def loss_spatial(org, enh):
    o = F.avg_pool2d(org.mean(1, keepdim=True), 4)
    e = F.avg_pool2d(enh.mean(1, keepdim=True), 4)
    loss = 0
    for dy, dx in ((0, 1), (1, 0)):
        do = o - torch.roll(o, (dy, dx), (2, 3))
        de = e - torch.roll(e, (dy, dx), (2, 3))
        loss = loss + ((do - de) ** 2).mean()
    return loss


def loss_tv(r):
    return ((r[:, :, 1:] - r[:, :, :-1]) ** 2).mean() + ((r[:, :, :, 1:] - r[:, :, :, :-1]) ** 2).mean()


# ------------------------------ data ----------------------------------------
def load_official(train_dirs, val_frac, seed=0):
    """official dataset: hold back val_frac of the training pairs for validation"""
    from src.datasets import find_pairs
    files = [pair for d in train_dirs for pair in find_pairs(d)]
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/official_val.json").write_text(json.dumps(
        [[str(a), str(b)] for a, b in val_files], indent=1))
    # kept as 8-bit in memory (4x less RAM than float), converted per batch
    u8 = lambda x: (x * 255 + 0.5).astype(np.uint8)
    train = [(u8(read_rgb(a, 512)), u8(read_rgb(b, 512))) for a, b in train_files]
    val = [(read_rgb(a, 256), read_rgb(b, 256)) for a, b in val_files]
    return train, val


def load_training_set():
    pairs, unpaired = [], []
    for cond in ("dark", "extreme", "nonuniform"):
        for low in sorted((ROOT / "data/synth/train" / cond / "low").glob("*.png")):
            pairs.append((read_rgb(low, 512), read_rgb(low.parents[1] / "high" / low.name, 512)))
    splits = json.loads((ROOT / "data/splits.json").read_text())
    for p in splits["real"]["train"]:
        unpaired.append(read_rgb(ROOT / p, 512))
    return pairs, unpaired


def random_crop(low, high, size):
    h, w = low.shape[:2]
    y, x = random.randint(0, h - size), random.randint(0, w - size)
    low = low[y:y + size, x:x + size]
    high = None if high is None else high[y:y + size, x:x + size]
    if random.random() < 0.5:
        low = low[:, ::-1]
        high = None if high is None else high[:, ::-1]
    return np.ascontiguousarray(low), None if high is None else np.ascontiguousarray(high)


def to_t(arrs):
    x = np.stack(arrs)
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    return t / 255.0 if x.dtype == np.uint8 else t


@torch.no_grad()
def validate(model, val_pairs=None):
    model.eval()
    scores = []
    if val_pairs is not None:
        for lo, hi in val_pairs:
            _, out, _ = model(to_t([lo]))
            scores.append(psnr(out[0].permute(1, 2, 0).clamp(0, 1).numpy(), hi))
        model.train()
        return float(np.mean(scores))
    for cond in ("dark", "extreme", "nonuniform"):
        for low in sorted((ROOT / "data/synth/val" / cond / "low").glob("*.png")):
            lo = read_rgb(low, 256)
            hi = read_rgb(low.parents[1] / "high" / low.name, 256)
            _, out, _ = model(to_t([lo]))
            scores.append(psnr(out[0].permute(1, 2, 0).clamp(0, 1).numpy(), hi))
    model.train()
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train_dir", action="append", default=None,
                    help="official training folder with low/high (can repeat). "
                         "If not given, our synthetic data is used.")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out", default="weights/zerodce_finetuned.pth")
    ap.add_argument("--l1_weight", type=float, default=20.0)
    ap.add_argument("--exp_target", type=float, default=0.5)
    ap.add_argument("--exp_weight", type=float, default=5.0)
    args = ap.parse_args()

    model = ZeroDCE()
    model.load_state_dict(torch.load(ROOT / "weights/zerodce_pretrained.pth", map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    val_pairs = None
    if args.train_dir:
        pairs, val_pairs = load_official(args.train_dir, args.val_frac)
        unpaired = []
        print(f"official data: {len(pairs)} train pairs, {len(val_pairs)} validation pairs "
              f"(list saved in data/official_val.json)")
    else:
        pairs, unpaired = load_training_set()
    print(f"{len(pairs)} paired + {len(unpaired)} unpaired training images")
    out_path = ROOT / args.out
    items = [(lo, hi) for lo, hi in pairs] + [(u, None) for u in unpaired]

    best = validate(model, val_pairs)
    log = [{"epoch": 0, "val_psnr": best}]
    print(f"epoch 0 (pretrained)  val PSNR {best:.2f}")
    torch.save(model.state_dict(), out_path)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        random.shuffle(items)
        losses = []
        for i in range(0, len(items) - args.batch + 1, args.batch):
            batch = [random_crop(lo, hi, args.crop) for lo, hi in items[i:i + args.batch]]
            x = to_t([b[0] for b in batch])
            _, out, r = model(x)
            loss = (loss_spatial(x, out) + args.exp_weight * loss_exposure(out, target=args.exp_target) + 5 * loss_color(out)
                    + 200 * loss_tv(r))
            paired = [k for k, b in enumerate(batch) if b[1] is not None]
            if paired:
                gt = to_t([batch[k][1] for k in paired])
                loss = loss + args.l1_weight * F.l1_loss(out[paired], gt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            losses.append(loss.item())
        v = validate(model, val_pairs)
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_psnr": v})
        flag = ""
        if v > best:
            best, flag = v, "  <- saved"
            torch.save(model.state_dict(), out_path)
        print(f"epoch {ep:2d}  loss {np.mean(losses):.4f}  val PSNR {v:.2f}  ({time.time() - t0:.0f}s){flag}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/finetune_log_{out_path.stem}.json").write_text(json.dumps(log, indent=1))
    print(f"best val PSNR {best:.2f}  (pretrained was {log[0]['val_psnr']:.2f})")


if __name__ == "__main__":
    main()
