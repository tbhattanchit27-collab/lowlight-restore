"""
Builds our train / val / test sets.

1) Paired data (synthetic). We take clean, normally exposed photos and make them
   dark the way a camera would: go to linear light, scale down, add a colour
   cast, add shot + read noise, go back to sRGB and save as 8-bit.
   Three lighting conditions, so we can test generalisation:
       dark        - normal night photo
       extreme     - almost black
       nonuniform  - one side lit, other side dark (like a street lamp)

   train : CBSD68 images 0-47      val : CBSD68 images 48-67
   test  : Kodak24 + McMaster (never seen in training or tuning)

2) Real dark photos (no ground truth) are split by name:
   DICM -> 44 train / 10 val / 10 test,  LIME + RetinexNet samples -> test only.

If you have the LOL dataset, you do not need this script for LOL: the evaluation
script reads LOL directly (--lol path/to/LOL/eval15).

usage:  python scripts/make_dataset.py
"""
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import read_rgb, save_rgb  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "synth"


def darken(img, condition, rng):
    lin = img ** 2.2
    h, w = img.shape[:2]
    if condition == "dark":
        light = rng.uniform(0.02, 0.05)
        mask = np.full((h, w), light, np.float32)
    elif condition == "extreme":
        light = rng.uniform(0.004, 0.012)
        mask = np.full((h, w), light, np.float32)
    else:  # nonuniform: a lamp somewhere, light falls off with distance
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / np.hypot(h, w)
        mask = (0.004 + 0.15 * np.exp(-(d / 0.35) ** 2)).astype(np.float32)

    cast = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    lin = lin * mask[..., None] * cast

    shot = rng.uniform(0.0002, 0.001)
    read = rng.uniform(0.0002, 0.0008)
    sigma = np.sqrt(shot * lin + read ** 2)
    lin = lin + rng.normal(0, 1, lin.shape).astype(np.float32) * sigma
    low = np.clip(lin, 0, 1) ** (1 / 2.2)
    return np.round(low * 255) / 255


def split_clean():
    cbsd = sorted((RAW / "CBSD68").glob("*.png"))
    test = sorted((RAW / "Kodak24").glob("*.png")) + sorted((RAW / "McMaster").glob("*.*"))
    return {"train": cbsd[:48], "val": cbsd[48:], "test": test}


def split_real():
    dicm = sorted((RAW / "real_dark" / "DICM").glob("*"))
    random.Random(0).shuffle(dicm)
    other = sorted((RAW / "real_dark" / "LIME").glob("*")) + sorted((RAW / "real_dark" / "RetinexNet").glob("*"))
    return {"train": dicm[:44], "val": dicm[44:54], "test": dicm[54:] + other}


def main():
    splits = split_clean()
    count = 0
    for split, files in splits.items():
        for cond in ("dark", "extreme", "nonuniform"):
            (OUT / split / cond / "low").mkdir(parents=True, exist_ok=True)
            (OUT / split / cond / "high").mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            img = read_rgb(f)
            for k, cond in enumerate(("dark", "extreme", "nonuniform")):
                rng = np.random.default_rng(1000 * k + i + {"train": 0, "val": 300, "test": 600}[split])
                name = f"{f.parent.name}_{f.stem}.png"
                save_rgb(OUT / split / cond / "high" / name, img)
                save_rgb(OUT / split / cond / "low" / name, darken(img, cond, rng))
                count += 1
    real = {k: [str(p.relative_to(ROOT)) for p in v] for k, v in split_real().items()}
    (ROOT / "data" / "splits.json").write_text(json.dumps(
        {"synthetic": {k: [p.name for p in v] for k, v in splits.items()}, "real": real}, indent=1))
    print(f"made {count} synthetic pairs, split file written to data/splits.json")
    print({k: len(v) for k, v in real.items()}, "real dark photos per split")


if __name__ == "__main__":
    main()
