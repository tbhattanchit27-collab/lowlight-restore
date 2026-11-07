"""
Deep learning part: Zero-DCE (Guo et al., CVPR 2020).

The network does not output an image. It outputs 8 "curve maps" (24 channels =
8 x RGB). Each curve is applied as   x = x + r * (x^2 - x)   which can only
brighten or darken smoothly and always stays in 0..1. That is why Zero-DCE
rarely creates strange colours.

Network code and original weights: https://github.com/Li-Chongyi/Zero-DCE
(non-commercial licence). The class below keeps the same layer names so the
original .pth file loads directly.

Our changes:
  * curves are predicted on a small copy (max side 512) and then upsampled and
    applied on the full image -> much faster on CPU, same look
  * quality aware output: we compare the 4-curve and 8-curve results and keep
    the one with better exposure / less noise (pick_by_quality)
  * fine-tuned on our own training split (scripts/finetune_zerodce.py)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# without this, tiny "denormal" float values make CPU convolutions ~30x slower
# (we measured 13.6 s -> 0.37 s for one 512 px image)
torch.set_flush_denormal(True)

class ZeroDCE(nn.Module):
    def __init__(self, nf=32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)

    def curves(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    def forward(self, x):
        r = self.curves(x)
        return apply_curves(x, r)


def apply_curves(x, r):
    """returns (output after 4 curves, output after 8 curves, curve maps)"""
    rs = torch.split(r, 3, dim=1)
    half = None
    for i, ri in enumerate(rs):
        x = x + ri * (x * x - x)
        if i == 3:
            half = x
    return half, x, r


_MODELS = {}


def load_model(path):
    if path not in _MODELS:
        m = ZeroDCE()
        state = torch.load(path, map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(state)
        m.eval()
        _MODELS[path] = m
    return _MODELS[path]


def default_weights():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ft = os.path.join(here, "weights", "zerodce_finetuned.pth")
    pre = os.path.join(here, "weights", "zerodce_pretrained.pth")
    return ft if os.path.exists(ft) else pre


@torch.no_grad()
def enhance_zerodce(img, weights=None, max_side=512, quality_aware=True):
    """img: float32 RGB 0..1 (H, W, 3). returns output and info dict."""
    model = load_model(weights or default_weights())
    h, w = img.shape[:2]
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float()

    scale = min(1.0, max_side / max(h, w))
    xs = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False) if scale < 1 else x
    r = model.curves(xs)
    if scale < 1:
        r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)
    half, full, _ = apply_curves(x, r)

    to_np = lambda t: t[0].permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
    half, full = to_np(half), to_np(full)

    if not quality_aware:
        return full, {"chosen": "8 curves"}
    out, name = pick_by_quality({"4 curves": half, "8 curves": full})
    return out, {"chosen": name}


def exposure_score(img):
    """lower is better. distance of mean brightness from 0.5 + clipped pixels."""
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return abs(float(y.mean()) - 0.5) + 2.0 * float((y > 0.97).mean())


def pick_by_quality(candidates):
    best = min(candidates, key=lambda k: exposure_score(candidates[k]))
    return candidates[best], best
