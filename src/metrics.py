"""
Quality metrics.

with a reference (paired data):   PSNR, SSIM, LPIPS, colour error (Delta E 2000),
                                  edge preservation (gradient correlation)
no reference (real dark photos):  NIQE, mean brightness, % over-exposed pixels,
                                  residual noise estimate
always:                           latency (measured in the evaluation script)

NIQE is a small port of the BasicSR implementation (Apache-2.0,
github.com/XPixelGroup/BasicSR). Only change: the half-size resize uses OpenCV
area resize instead of a MATLAB clone, so values differ slightly (~0.1) from
MATLAB. We use the same function for every method, so the comparison is fair.
"""
import math
import os

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma as G
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma


def psnr(out, ref):
    return float(peak_signal_noise_ratio(ref, out, data_range=1.0))


def ssim(out, ref):
    return float(structural_similarity(ref, out, channel_axis=-1, data_range=1.0))


def delta_e(out, ref):
    return float(deltaE_ciede2000(rgb2lab(ref), rgb2lab(out)).mean())


def edge_preservation(out, ref):
    """correlation between Sobel gradient magnitudes. 1.0 = edges identical."""
    def grad(x):
        y = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)).ravel()
    a, b = grad(out), grad(ref)
    return float(np.corrcoef(a, b)[0, 1])


def noise_level(img):
    return float(np.mean(estimate_sigma(img, channel_axis=-1)))


def overexposed(img, thr=0.98):
    return float((img.max(axis=2) > thr).mean() * 100)


# ------------------------------- LPIPS --------------------------------------
_lpips = None


def lpips_score(out, ref):
    """returns None if the AlexNet weights can not be downloaded (offline)."""
    global _lpips
    if _lpips is False:          # already failed once, do not retry the download
        return None
    try:
        import torch
        import lpips
        if _lpips is None:
            _lpips = lpips.LPIPS(net="alex", verbose=False)
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float() * 2 - 1
        with torch.no_grad():
            return float(_lpips(t(out), t(ref)).item())
    except Exception:
        _lpips = False
        return None


# -------------------------------- NIQE --------------------------------------
_P = np.load(os.path.join(os.path.dirname(__file__), "niqe_pris_params.npz"))
_GAM = np.arange(0.2, 10.001, 0.001)
_R_GAM = np.square(G(2 / _GAM)) / (G(1 / _GAM) * G(3 / _GAM))


def _aggd(block):
    block = block.ravel()
    left = np.sqrt(np.mean(block[block < 0] ** 2))
    right = np.sqrt(np.mean(block[block > 0] ** 2))
    gh = left / right
    rhat = np.mean(np.abs(block)) ** 2 / np.mean(block ** 2)
    rn = rhat * (gh ** 3 + 1) * (gh + 1) / (gh ** 2 + 1) ** 2
    a = _GAM[np.argmin((_R_GAM - rn) ** 2)]
    k = np.sqrt(G(1 / a) / G(3 / a))
    return a, left * k, right * k


def _features(block):
    a, bl, br = _aggd(block)
    f = [a, (bl + br) / 2]
    for sh in ([0, 1], [1, 0], [1, 1], [1, -1]):
        a, bl, br = _aggd(block * np.roll(block, sh, axis=(0, 1)))
        f += [a, (br - bl) * G(2 / a) / G(1 / a), bl, br]
    return f


def niqe(img, bs=96):
    """img: float RGB 0..1. lower = more natural looking."""
    y = 16 + 65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]
    y = y.astype(np.float64)
    h, w = y.shape
    nh, nw = h // bs, w // bs
    if nh < 1 or nw < 1:
        return float("nan")
    y = y[:nh * bs, :nw * bs]
    win = _P["gaussian_window"]
    feats = []
    for scale in (1, 2):
        mu = convolve(y, win, mode="nearest")
        sig = np.sqrt(np.abs(convolve(y * y, win, mode="nearest") - mu * mu))
        yn = (y - mu) / (sig + 1)
        b = bs // scale
        feats.append(np.array([_features(yn[i * b:(i + 1) * b, j * b:(j + 1) * b])
                               for j in range(nw) for i in range(nh)]))
        if scale == 1:
            y = cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA)
    d = np.concatenate(feats, axis=1)
    mu_d = np.nanmean(d, axis=0)
    cov_d = np.cov(d[~np.isnan(d).any(axis=1)], rowvar=False)
    diff = _P["mu_pris_param"] - mu_d
    inv = np.linalg.pinv((_P["cov_pris_param"] + cov_d) / 2)
    return float(np.sqrt(np.squeeze(diff @ inv @ diff.T)))
