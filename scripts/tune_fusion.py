"""
Chooses the fusion rule of the hybrid pipeline using the VALIDATION set only
(data/synth/val, 60 pairs). The test set is not touched.

usage: python scripts/tune_fusion.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as M  # noqa: E402
from src.classical import enhance_classical  # noqa: E402
from src.hybrid import enhance_hybrid  # noqa: E402
from src.io_utils import read_rgb  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--official", action="store_true",
                help="use the validation pairs held back by finetune_zerodce.py --train_dir")
args = ap.parse_args()

pairs = []
if args.official:
    for a, b in json.loads((ROOT / "data/official_val.json").read_text()):
        pairs.append((read_rgb(a), read_rgb(b)))
else:
    for cond in ("dark", "extreme", "nonuniform"):
        for f in sorted((ROOT / "data/synth/val" / cond / "low").glob("*.png")):
            pairs.append((read_rgb(f), read_rgb(f.parents[1] / "high" / f.name)))


scores = {}


def report(name, fn, t=None):
    outs = [(fn(lo), hi) for lo, hi in pairs]
    ps, ss = np.mean([M.psnr(o, h) for o, h in outs]), np.mean([M.ssim(o, h) for o, h in outs])
    if t is not None:
        scores[t] = ps + 10 * ss          # one number that weighs PSNR and SSIM about equally
    line = f"{name:38s} PSNR {ps:6.2f}  SSIM {ss:.4f}"
    print(line)
    return line


lines = [report("classical only", lambda x: enhance_classical(x)[0]),
         report("hybrid, DL branch only (no fusion)", lambda x: enhance_hybrid(x, {"fusion": False})[0]),
         report("hybrid, fixed 50/50 average", lambda x: enhance_hybrid(x, {"fusion": "average"})[0])]
for t in (0.2, 0.3, 0.4, 0.6):
    lines.append(report(f"hybrid, weight from T, t_high={t}", lambda x, t=t: enhance_hybrid(x, t_high=t)[0], t))
name = "official" if args.official else "synth"
(ROOT / f"results/tune_fusion_{name}.txt").write_text(f"validation set ({name}, {len(pairs)} pairs)\n" + "\n".join(lines) + "\n")

best_t = max(scores, key=scores.get)
print(f"chosen t_high = {best_t} (best PSNR + 10 x SSIM on validation)")

# second step: CLAHE strength, with the chosen t_high (0 = CLAHE off)
clahe_scores = {}
for cs in (0.0, 0.35, 0.7):
    scores.clear()
    lines.append(report(f"hybrid t_high={best_t}, CLAHE scale {cs}",
                        lambda x, cs=cs: enhance_hybrid(x, t_high=best_t, clahe_scale=cs)[0], cs))
    clahe_scores[cs] = scores[cs]
best_cs = max(clahe_scores, key=clahe_scores.get)
print(f"chosen CLAHE scale = {best_cs} (0 means CLAHE off)")
lines.append(f"chosen: t_high={best_t}, clahe_scale={best_cs}")
(ROOT / f"results/tune_fusion_{name}.txt").write_text(f"validation set ({name}, {len(pairs)} pairs)\n" + "\n".join(lines) + "\n")
if args.official:
    (ROOT / "weights/fusion_config.json").write_text(json.dumps({"t_high": best_t, "clahe_scale": best_cs}))
    print("saved to weights/fusion_config.json (the app and evaluation use it automatically)")
