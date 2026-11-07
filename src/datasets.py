"""
Finds (low, high) image pairs in the official datasets, whatever the folder names.

Works with:
  LOL v1   : our485/low + our485/high,  eval15/low + eval15/high      (same file names)
  LOL-v2   : Real_captured/Train/Low + Normal   (low00001.png <-> normal00001.png)
             Synthetic/Train/Low + Normal       (same file names)
  our synthetic data: data/synth/<split>/<cond>/low + high

Pass the folder that directly contains the low and high folders.
"""
import re
from pathlib import Path

LOW_NAMES = {"low", "input", "dark"}
HIGH_NAMES = {"high", "normal", "gt", "target", "expertc"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


def _key(p):
    """file number used to match low00012.png with normal00012.png"""
    nums = re.findall(r"\d+", p.stem)
    if not nums:
        return p.stem.lower()
    return nums[-1].lstrip("0") or "0"


def find_pairs(root):
    root = Path(root)
    subs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    low_dir = next((subs[n] for n in subs if n in LOW_NAMES), None)
    high_dir = next((subs[n] for n in subs if n in HIGH_NAMES), None)
    if low_dir is None or high_dir is None:
        raise FileNotFoundError(
            f"{root} must contain a low folder ({'/'.join(sorted(LOW_NAMES))}) and a high folder "
            f"({'/'.join(sorted(HIGH_NAMES))}). Found: {sorted(subs)}")
    highs = {_key(p): p for p in _images(high_dir)}
    pairs = [(lo, highs[_key(lo)]) for lo in _images(low_dir) if _key(lo) in highs]
    if not pairs:
        raise FileNotFoundError(f"no matching image pairs in {low_dir} and {high_dir}")
    return pairs
