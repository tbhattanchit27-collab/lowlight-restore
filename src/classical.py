"""
Classical low-light enhancement.

Steps (each one can be switched off for the ablation study):
  1. look at the image and pick parameters        -> analyse_image()
  2. estimate the illumination map                -> estimate_illumination()
  3. Retinex style brightening  I * T^(gamma-1)   -> retinex_enhance()
  4. edge preserving denoising (bilateral)        -> denoise()
  5. CLAHE on the L channel of LAB                -> apply_clahe()
  6. white balance + highlight protection         -> colour_correct()

All images are float32 RGB in the range 0..1.
"""
import cv2
import numpy as np
from skimage.restoration import estimate_sigma


# ---------------------------------------------------------------------------
# guided filter (He et al. 2010) - written with box filters so we do not need
# opencv-contrib. guide and src are 2D float images.
# ---------------------------------------------------------------------------
def guided_filter(guide, src, radius=15, eps=1e-3):
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean = lambda x: cv2.boxFilter(x, -1, ksize)
    mean_i = mean(guide)
    mean_p = mean(src)
    cov_ip = mean(guide * src) - mean_i * mean_p
    var_i = mean(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return mean(a) * guide + mean(b)


def luminance(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# ---------------------------------------------------------------------------
# 1. image analysis -> parameters. this is the "adaptive" part. nothing in the
#    pipeline uses a fixed strength, everything is read from these numbers.
# ---------------------------------------------------------------------------
def analyse_image(img):
    y = luminance(img)
    mean_lum = float(y.mean())
    dark_ratio = float((y < 0.1).mean())          # how much of the image is very dark

    # noise: wavelet based estimate on a small copy (fast enough)
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) if max(img.shape[:2]) > 800 else img
    noise_sigma = float(np.mean(estimate_sigma(small, channel_axis=-1)))

    # colour cast: how far are the channel means from grey (grey-world idea)
    ch_mean = img.reshape(-1, 3).mean(0) + 1e-6
    colour_cast = float(np.abs(ch_mean / ch_mean.mean() - 1).max())

    # non-uniform light: spread of a heavily blurred luminance
    blur = cv2.GaussianBlur(y, (0, 0), sigmaX=max(img.shape[:2]) / 20)
    nonuniform = float(blur.std() / (blur.mean() + 1e-3))

    contrast = float(np.percentile(y, 95) - np.percentile(y, 5))

    stats = dict(mean_lum=mean_lum, dark_ratio=dark_ratio, noise_sigma=noise_sigma,
                 colour_cast=colour_cast, nonuniform=nonuniform, contrast=contrast)
    stats["params"] = choose_params(stats)
    return stats


def choose_params(s):
    # target brightness of the illumination map. darker image -> we aim a bit lower,
    # because pushing an extremely dark image to 0.5 just shows the noise.
    target = 0.45 if s["mean_lum"] > 0.05 else 0.38

    # noise is amplified roughly by (target / mean), so scale denoise with that gain
    gain = np.clip(target / max(s["mean_lum"], 1e-3), 1, 20)
    denoise_strength = float(np.clip(s["noise_sigma"] * gain * 6, 0.0, 1.0))

    # CLAHE: less clipping when noisy (CLAHE also boosts noise), more when flat
    clip = 2.0 if s["contrast"] < 0.25 else 1.5
    clip = clip * (1 - 0.5 * denoise_strength)

    # white balance only when the cast is clearly strong, and never fully.
    # grey-world wrongly "fixes" real coloured scenes (sunsets, neon), we saw this
    # in our first test run, so the strength is capped at 0.3
    wb_strength = float(np.clip((s["colour_cast"] - 0.2) * 0.6, 0.0, 0.3))

    return dict(target=target, denoise_strength=denoise_strength,
                clahe_clip=float(clip), wb_strength=wb_strength)


# ---------------------------------------------------------------------------
# 2. illumination map  (LIME style: max of RGB, then smoothed with a guided
#    filter so edges between bright and dark objects stay sharp -> no halos)
# ---------------------------------------------------------------------------
def estimate_illumination(img, radius=None):
    t0 = img.max(axis=2)
    if radius is None:
        radius = max(5, int(min(img.shape[:2]) / 40))
    t = guided_filter(luminance(img), t0, radius=radius, eps=1e-3)
    return np.clip(t, 1e-3, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Retinex enhancement.  I = R * T  ->  output = R * T^gamma = I * T^(gamma-1)
#    gamma is picked so that the mean of the new illumination hits the target.
# ---------------------------------------------------------------------------
def retinex_enhance(img, illum, target=0.45):
    m = float(np.clip(illum.mean(), 1e-3, 0.99))
    gamma = float(np.clip(np.log(target) / np.log(m), 0.2, 1.0))
    t = np.maximum(illum, 0.02)               # floor stops black pixels exploding
    out = img * (t ** (gamma - 1.0))[..., None]
    return np.clip(out, 0, 1), gamma


# ---------------------------------------------------------------------------
# 4. denoising. bilateral filter in YCrCb (chroma noise is the ugly part so it
#    gets a stronger filter). the result is blended back using the illumination
#    map: dark regions were amplified most, so they get the most denoising.
# ---------------------------------------------------------------------------
def denoise(img, strength, illum=None):
    if strength < 0.02:
        return img
    u8 = (img * 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    s_y = 10 + 60 * strength
    s_c = 20 + 100 * strength
    y = cv2.bilateralFilter(ycc[..., 0], 7, s_y, 5)
    cr = cv2.bilateralFilter(ycc[..., 1], 9, s_c, 7)
    cb = cv2.bilateralFilter(ycc[..., 2], 9, s_c, 7)
    den = cv2.cvtColor(np.dstack([y, cr, cb]), cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255
    if illum is None:
        return den
    w = np.clip(1.0 - illum / 0.5, 0.3, 1.0)[..., None]   # dark -> w=1
    return den * w + img * (1 - w)


# ---------------------------------------------------------------------------
# 5. CLAHE on L of LAB (only luminance, so colours do not shift)
# ---------------------------------------------------------------------------
def apply_clahe(img, clip=1.5, tiles=8):
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


# ---------------------------------------------------------------------------
# 6. colour correction: partial grey-world white balance + soft knee on the
#    highlights so bright lamps do not clip to pure white.
# ---------------------------------------------------------------------------
def colour_correct(img, wb_strength=0.5):
    ch_mean = img.reshape(-1, 3).mean(0) + 1e-6
    gains = ch_mean.mean() / ch_mean
    gains = 1 + wb_strength * (gains - 1)
    out = img * gains[None, None, :]
    return highlight_protect(out)


def highlight_protect(img, knee=0.85):
    # values above the knee are compressed smoothly instead of clipped
    over = np.maximum(img - knee, 0)
    out = np.where(img > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), img)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# full classical pipeline. `use` lets the ablation turn steps off.
# returns the output plus every intermediate image for the app.
# ---------------------------------------------------------------------------
DEFAULT_STEPS = dict(adaptive=True, retinex=True, denoise=True, clahe=True, colour=True)


def enhance_classical(img, use=None):
    use = {**DEFAULT_STEPS, **(use or {})}
    stats = analyse_image(img)
    p = stats["params"] if use["adaptive"] else \
        dict(target=0.45, denoise_strength=0.5, clahe_clip=2.0, wb_strength=0.5)

    steps = {}
    illum = estimate_illumination(img)
    steps["illumination"] = illum

    if use["retinex"]:
        out, gamma = retinex_enhance(img, illum, p["target"])
    else:  # plain global gamma, for comparison
        gamma = float(np.clip(np.log(p["target"]) / np.log(max(luminance(img).mean(), 1e-3)), 0.2, 1))
        out = np.clip(img ** gamma, 0, 1)
    steps["retinex"] = out

    if use["denoise"]:
        out = denoise(out, p["denoise_strength"], illum)
    steps["denoised"] = out

    if use["clahe"]:
        out = apply_clahe(out, p["clahe_clip"])
    steps["clahe"] = out

    if use["colour"]:
        out = colour_correct(out, p["wb_strength"])
    steps["colour"] = out

    info = {**{k: v for k, v in stats.items() if k != "params"}, **p, "gamma": gamma}
    return out.astype(np.float32), steps, info
