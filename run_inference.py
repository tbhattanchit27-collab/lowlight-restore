"""
Enhance every image in a folder.

usage:
  python run_inference.py --input_dir ./samples/ --output_dir ./results/
  python run_inference.py --input_dir ./samples/ --output_dir ./out/ --method classical
  python run_inference.py --input_dir ./samples/ --output_dir ./out/ --save_steps

--method      hybrid (default) | classical | dl
--save_steps  also save the illumination map, both branches and the fusion weight map
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.classical import enhance_classical
from src.hybrid import enhance_hybrid
from src.io_utils import read_rgb, save_rgb
from src.zerodce import enhance_zerodce

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def heat(gray):
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method", choices=["hybrid", "classical", "dl"], default="hybrid")
    ap.add_argument("--max_side", type=int, default=None, help="resize larger images (optional)")
    ap.add_argument("--save_steps", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images found in {args.input_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        img = read_rgb(f, args.max_side)
        t0 = time.perf_counter()
        if args.method == "hybrid":
            out, steps, _ = enhance_hybrid(img)
        elif args.method == "classical":
            out, steps, _ = enhance_classical(img)
        else:
            out, _ = enhance_zerodce(img)
            steps = {}
        ms = (time.perf_counter() - t0) * 1000
        save_rgb(out_dir / f"{f.stem}_{args.method}.png", out)
        if args.save_steps:
            for name, im in steps.items():
                im = heat(im) if im.ndim == 2 else im
                save_rgb(out_dir / f"{f.stem}_step_{name}.png", im)
        rows.append((f.name, img.shape[1], img.shape[0], round(ms, 1)))
        print(f"{f.name}: {ms:.0f} ms")

    with open(out_dir / "latency.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "width", "height", "ms"])
        w.writerows(rows)
    print(f"done: {len(files)} images -> {out_dir} (mean {np.mean([r[3] for r in rows]):.0f} ms per image)")


if __name__ == "__main__":
    main()
