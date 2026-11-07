"""
Ablation study: turn off one part of the pipeline at a time and see how much
the scores drop. Run on the paired synthetic test set (PSNR, SSIM, ...) and
on the real dark test photos (NIQE).

usage: python scripts/ablation.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as M  # noqa: E402
from src.classical import enhance_classical  # noqa: E402
from src.hybrid import enhance_hybrid  # noqa: E402
from src.io_utils import read_rgb  # noqa: E402

FT = str(ROOT / "weights/zerodce_finetuned.pth")
PRE = str(ROOT / "weights/zerodce_pretrained.pth")

VARIANTS = {
    "hybrid full": lambda x: enhance_hybrid(x, weights=FT)[0],
    "- adaptive params (fixed)": lambda x: enhance_hybrid(x, {"adaptive": False}, FT)[0],
    "- pre-gain": lambda x: enhance_hybrid(x, {"pregain": False}, FT)[0],
    "- fusion (DL branch only)": lambda x: enhance_hybrid(x, {"fusion": False}, FT)[0],
    "fusion = fixed 50/50 average": lambda x: enhance_hybrid(x, {"fusion": "average"}, FT)[0],
    "- denoise": lambda x: enhance_hybrid(x, {"denoise": False}, FT)[0],
    "- CLAHE": lambda x: enhance_hybrid(x, {"clahe": False}, FT)[0],
    "- colour correction": lambda x: enhance_hybrid(x, {"colour": False}, FT)[0],
    "- fine-tuning (pretrained weights)": lambda x: enhance_hybrid(x, weights=PRE)[0],
    "classical full": lambda x: enhance_classical(x)[0],
    "classical - Retinex (global gamma)": lambda x: enhance_classical(x, {"retinex": False})[0],
    "classical - adaptive params": lambda x: enhance_classical(x, {"adaptive": False})[0],
    "classical - denoise": lambda x: enhance_classical(x, {"denoise": False})[0],
    "classical - CLAHE": lambda x: enhance_classical(x, {"clahe": False})[0],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default=None, help="official test folder (low + high)")
    ap.add_argument("--name", default="synth")
    args = ap.parse_args()
    pairs = []
    if args.test_dir:
        from src.datasets import find_pairs
        pairs = [(read_rgb(a), read_rgb(b)) for a, b in find_pairs(args.test_dir)]
    else:
        for cond in ("dark", "extreme", "nonuniform"):
            for f in sorted((ROOT / "data/synth/test" / cond / "low").glob("*.png")):
                pairs.append((read_rgb(f), read_rgb(f.parents[1] / "high" / f.name)))
    real = [read_rgb(ROOT / p, 800) for p in json.loads((ROOT / "data/splits.json").read_text())["real"]["test"]]

    rows = []
    for name, fn in VARIANTS.items():
        outs = [(fn(lo), hi) for lo, hi in pairs]
        niqe = [M.niqe(fn(x)) for x in real]
        rows.append(dict(variant=name,
                         PSNR=sum(M.psnr(o, h) for o, h in outs) / len(outs),
                         SSIM=sum(M.ssim(o, h) for o, h in outs) / len(outs),
                         DeltaE=sum(M.delta_e(o, h) for o, h in outs) / len(outs),
                         EdgeCorr=sum(M.edge_preservation(o, h) for o, h in outs) / len(outs),
                         OverExp=sum(M.overexposed(o) for o, _ in outs) / len(outs),
                         NIQE_real=sum(niqe) / len(niqe)))
        print(rows[-1])
    df = pd.DataFrame(rows).round(4)
    df.to_csv(ROOT / f"results/ablation_{args.name}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
