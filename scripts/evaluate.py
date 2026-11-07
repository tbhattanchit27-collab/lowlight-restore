"""
Compares the methods on the test sets and writes results/ files.

  paired test  : data/synth/test (Kodak24 + McMaster, 3 lighting conditions)
                 or the LOL dataset with  --lol path/to/LOL/eval15
  unpaired test: real dark photos (DICM test split, LIME, RetinexNet samples)

usage:
  python scripts/evaluate.py                       # our test sets
  python scripts/evaluate.py --lol data/LOL/eval15 # LOL (folders low/ and high/)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as M  # noqa: E402
from src.classical import enhance_classical  # noqa: E402
from src.hybrid import enhance_hybrid  # noqa: E402
from src.io_utils import read_rgb, save_rgb  # noqa: E402
from src.zerodce import enhance_zerodce  # noqa: E402

PRE = str(ROOT / "weights/zerodce_pretrained.pth")
FT = str(ROOT / "weights/zerodce_finetuned.pth")

METHODS = {
    "input (no enhancement)": lambda x: x,
    "classical": lambda x: enhance_classical(x)[0],
    "DL Zero-DCE pretrained": lambda x: enhance_zerodce(x, weights=PRE)[0],
    "DL Zero-DCE fine-tuned": lambda x: enhance_zerodce(x, weights=FT)[0],
    "hybrid (ours)": lambda x: enhance_hybrid(x, weights=FT)[0],
}


def paired_files(args):
    if args.test_dir:
        from src.datasets import find_pairs
        return [(args.name, lo, hi) for lo, hi in find_pairs(args.test_dir)]
    out = []
    for cond in ("dark", "extreme", "nonuniform"):
        for f in sorted((ROOT / "data/synth/test" / cond / "low").glob("*.png")):
            out.append((cond, f, f.parents[1] / "high" / f.name))
    return out


def run_paired(files, save_dir):
    rows = []
    for i, (cond, low_p, high_p) in enumerate(files):
        low, high = read_rgb(low_p), read_rgb(high_p)
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            out = fn(low)
            ms = (time.perf_counter() - t0) * 1000
            rows.append(dict(condition=cond, image=low_p.name, method=name,
                             PSNR=M.psnr(out, high), SSIM=M.ssim(out, high),
                             LPIPS=M.lpips_score(out, high), DeltaE=M.delta_e(out, high),
                             EdgeCorr=M.edge_preservation(out, high),
                             Noise=M.noise_level(out), OverExp=M.overexposed(out), ms=ms))
            if i % 10 == 0:
                save_rgb(save_dir / f"{cond}_{low_p.stem}_{name.split()[0]}{'_ft' if 'fine' in name else ''}.png", out)
        if i % 20 == 0:
            print(f"  paired {i + 1}/{len(files)}")
    return pd.DataFrame(rows)


def run_unpaired(save_dir):
    splits = json.loads((ROOT / "data/splits.json").read_text())
    rows = []
    for i, p in enumerate(splits["real"]["test"]):
        img = read_rgb(ROOT / p, max_side=800)
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            out = fn(img)
            ms = (time.perf_counter() - t0) * 1000
            rows.append(dict(source=Path(p).parent.name, image=Path(p).name, method=name,
                             NIQE=M.niqe(out), Brightness=float(out.mean()),
                             Noise=M.noise_level(out), OverExp=M.overexposed(out), ms=ms))
            save_rgb(save_dir / f"{Path(p).parent.name}_{Path(p).stem}_{name.split()[0]}{'_ft' if 'fine' in name else ''}.png", out)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", "--lol", dest="test_dir", default=None,
                    help="official test folder containing low + high folders "
                         "(e.g. LOL-v2/Real_captured/Test or LOL/eval15)")
    ap.add_argument("--name", default=None, help="short name used for result files, e.g. lolv2_real")
    ap.add_argument("--skip_unpaired", action="store_true")
    args = ap.parse_args()
    res = ROOT / "results"
    tag = args.name or ("official" if args.test_dir else "synth")
    args.name = tag
    (res / f"images_{tag}").mkdir(parents=True, exist_ok=True)

    # warm-up so model loading is not counted in the latency
    for fn in METHODS.values():
        fn(np.random.rand(256, 256, 3).astype(np.float32) * 0.1)

    df = run_paired(paired_files(args), res / f"images_{tag}")
    df.to_csv(res / f"paired_{tag}_per_image.csv", index=False)
    cols = ["PSNR", "SSIM", "LPIPS", "DeltaE", "EdgeCorr", "Noise", "OverExp", "ms"]
    summary = df.groupby("method", sort=False)[cols].mean().round(4)
    by_cond = df.groupby(["condition", "method"], sort=False)[["PSNR", "SSIM"]].mean().round(3)
    summary.to_csv(res / f"paired_{tag}_summary.csv")
    by_cond.to_csv(res / f"paired_{tag}_by_condition.csv")
    print(summary.to_string())
    print(by_cond.to_string())

    if not args.skip_unpaired and not args.test_dir:
        (res / "images_real").mkdir(exist_ok=True)
        du = run_unpaired(res / "images_real")
        du.to_csv(res / "unpaired_real_per_image.csv", index=False)
        us = du.groupby("method", sort=False)[["NIQE", "Brightness", "Noise", "OverExp", "ms"]].mean().round(4)
        us.to_csv(res / "unpaired_real_summary.csv")
        print(us.to_string())


if __name__ == "__main__":
    main()
