"""
Hybrid pipeline = classical illumination estimation + Zero-DCE + adaptive fusion.

  input
    |-- analyse_image()  -> stats and parameters (adaptive)
    |-- estimate_illumination() -> T
    |
    |-- [pre-gain]  if the image is extremely dark, lift it a little with the
    |               illumination map first, so Zero-DCE gets an input that looks
    |               like what it was trained on
    |
    |-- branch A: Retinex output (classical)
    |-- branch B: Zero-DCE output (deep learning)
    |
    |-- fusion: per-pixel weight from the illumination map T.
    |           very dark areas -> trust Retinex, lit areas -> trust Zero-DCE
    |
    |-- denoise (strength from noise estimate, more in dark areas of T)
    |-- light CLAHE
    |-- white balance + highlight protection
  output

How we picked the fusion rule (all on the VALIDATION set, data/synth/val):
  exposure-fusion weights (Mertens)   16.72 dB   <- weights followed noise
  plain 50/50 average                 17.24 dB
  inverse-noise weighting             17.26 dB
  weight from illumination T (ours)   17.48 dB
see scripts/tune_fusion.py and results/tune_fusion.txt
"""
import json
from pathlib import Path

import cv2
import numpy as np

from . import classical as C
from .zerodce import enhance_zerodce

DEFAULT_STEPS = dict(adaptive=True, pregain=True, fusion=True, denoise=True, clahe=True, colour=True)

# DL weight = T / FUSION_T_HIGH (clipped to 0..1), then blurred so there are no seams
FUSION_T_HIGH = 0.3
FUSION_BLUR = 5

# CLAHE strength relative to the classical pipeline (0 = CLAHE off)
CLAHE_SCALE = 0.7

# if the settings were re-tuned on an official validation set, use those values
_cfg = Path(__file__).resolve().parents[1] / "weights" / "fusion_config.json"
if _cfg.exists():
    _c = json.loads(_cfg.read_text())
    FUSION_T_HIGH = float(_c.get("t_high", FUSION_T_HIGH))
    CLAHE_SCALE = float(_c.get("clahe_scale", CLAHE_SCALE))


def fusion_weight_dl(illum, t_high=FUSION_T_HIGH):
    w = np.clip(illum / t_high, 0, 1).astype(np.float32)
    return cv2.GaussianBlur(w, (0, 0), FUSION_BLUR)


def enhance_hybrid(img, use=None, weights=None, t_high=None, clahe_scale=None):
    t_high = t_high or FUSION_T_HIGH
    clahe_scale = CLAHE_SCALE if clahe_scale is None else clahe_scale
    use = {**DEFAULT_STEPS, **(use or {})}
    stats = C.analyse_image(img)
    p = stats["params"] if use["adaptive"] else \
        dict(target=0.45, denoise_strength=0.5, clahe_clip=2.0, wb_strength=0.5)
    steps, info = {}, {}

    illum = C.estimate_illumination(img)
    steps["illumination"] = illum

    # pre-gain for very dark inputs
    dl_input = img
    if use["pregain"] and stats["mean_lum"] < 0.10:
        dl_input, g = C.retinex_enhance(img, illum, target=0.20)
        info["pregain_gamma"] = g
    steps["dl_input"] = dl_input

    # two branches
    retinex_out, gamma = C.retinex_enhance(img, illum, p["target"])
    dl_out, dl_info = enhance_zerodce(dl_input, weights=weights)
    steps["classical_branch"] = retinex_out
    steps["dl_branch"] = dl_out
    info.update(dl_info)

    # adaptive fusion
    if use["fusion"] == "average":            # only used in the ablation
        out = 0.5 * retinex_out + 0.5 * dl_out
    elif use["fusion"]:
        w = fusion_weight_dl(illum, t_high)[..., None]
        out = w * dl_out + (1 - w) * retinex_out
        info["dl_weight_share"] = float(w.mean())
        steps["fusion_weight_dl"] = w[..., 0]
    else:
        out = dl_out
    steps["fused"] = out

    if use["denoise"]:
        out = C.denoise(out, p["denoise_strength"], illum)
    steps["denoised"] = out

    if use["clahe"] and clahe_scale > 0:
        out = C.apply_clahe(out, p["clahe_clip"] * clahe_scale)  # lighter: both branches already lift contrast
    steps["clahe"] = out

    if use["colour"]:
        out = C.colour_correct(out, p["wb_strength"])
    steps["final"] = out

    info.update({k: v for k, v in stats.items() if k != "params"})
    info.update(p)
    info["gamma"] = gamma
    return out.astype(np.float32), steps, info
