% Robust Low-Light Image Restoration and Enhancement (P1)
% Team TXX: Name1 (USN1), Name2 (USN2), Name3 (USN3), Name4 (USN4)
% Computer Vision Hackathon 2026

## 1. Problem

Photos taken at night or in dark rooms are dark, noisy and often have a colour cast.
If we just increase brightness, the noise also gets bigger, bright lamps become pure
white and colours look wrong. The task is to build a system that brightens the image
"intelligently": it should look at each image, decide how much to brighten, remove
noise and fix colour, without halos or over-exposure.

## 2. Our idea in one line

Run a classical Retinex method and a small deep learning model (Zero-DCE) side by side,
then mix their results pixel by pixel using the illumination map: the darkest areas take
the Retinex result, the lit areas take Zero-DCE. All strengths come from measurements on
the input image.

![architecture](docs/architecture.png)

## 3. Pipeline

### 3.1 Image analysis (adaptive parameters)
We measure six numbers from the input:

| Measure | How | Used for |
|---|---|---|
| mean brightness | mean of luminance | brightening target, pre-gain on/off |
| dark ratio | % of pixels below 0.1 | reported in app |
| noise sigma | wavelet estimate (`skimage.estimate_sigma`) | denoise strength, CLAHE clip |
| colour cast | how far R, G, B means are from grey | white balance strength |
| non-uniformity | std / mean of a very blurred luminance | reported in app |
| contrast | 95th minus 5th percentile | CLAHE clip |

Noise gets amplified roughly by `target / mean brightness`, so the denoise strength is
`noise sigma x that gain x 6` (clipped to 0..1). This way a clean, bright image gets
almost no smoothing and a dark, noisy one gets strong smoothing.

### 3.2 Illumination estimation (classical)
Retinex theory says an image = reflectance (the real colours) x illumination (the light).
We estimate the illumination T like the LIME paper: take max(R, G, B) for every pixel,
then smooth it with a guided filter that uses the image itself as the guide. The guided
filter keeps edges between bright and dark objects, which is what stops halos.
We wrote the guided filter ourselves with box filters (15 lines), so no extra library.

### 3.3 Retinex enhancement (classical branch)
We change the light, not the colours: `output = I x T^(gamma - 1)`.
gamma is chosen so the mean of the new light `T^gamma` hits a target (0.45, or 0.38 for
extremely dark photos so we do not just show noise). A floor of 0.02 on T stops black
pixels from blowing up.

### 3.4 Zero-DCE (deep learning branch)
Zero-DCE (CVPR 2020) predicts 8 per-pixel curves instead of an image. Each curve is
`x + r(x^2 - x)`, which is smooth and stays inside 0..1, so colours rarely go strange.
The model has only 79,416 parameters. What we added:

* **Fine-tuning** on our training split (Section 5). Validation PSNR went from 16.28 dB
  (original weights) to 17.68 dB.
* **Fast inference**: curves are predicted on a copy resized to max 512 px, then
  upsampled and applied on the full image.
* **Quality-aware output**: the model gives a result after 4 curves and after 8 curves.
  We keep the one whose brightness is closest to 0.5 with the fewest clipped pixels.
* **CPU fix**: `torch.set_flush_denormal(True)` made one 512 px image go from 13.6 s to
  0.37 s on our CPU. Without it the app would be too slow.

### 3.5 Pre-gain (hybrid)
Zero-DCE was trained on images that are dark but not black. When the mean brightness is
below 0.10 we first lift the image a little with the Retinex map (target 0.20) and give
that to Zero-DCE. This is where classical illumination estimation directly helps the
deep model.

### 3.6 Adaptive fusion (hybrid)
Each pixel gets a Zero-DCE weight `w = clip(T / 0.3, 0, 1)`, smoothed with a Gaussian
(sigma 5) so there are no seams. Output = `w x ZeroDCE + (1 - w) x Retinex`.
So in the darkest parts of the photo (T near 0) we trust the Retinex branch, and in parts
that already have some light we trust Zero-DCE. The app shows this weight map.

We did not guess this rule. We tried four rules on the **validation** set only:

| Fusion rule (validation, 60 pairs) | PSNR (dB) | SSIM |
|---|---|---|
| exposure-fusion weights (Mertens), our first try | 16.72 | 0.579 |
| fixed 50/50 average | 17.24 | 0.596 |
| inverse-noise weighting | 17.26 | 0.598 |
| **weight from illumination T (t = 0.3)** | **17.48** | **0.614** |

The first try (well-exposedness x contrast weights) did worst because Zero-DCE's output
is noisier, and noise looks like "contrast", so the weights went to the noisy branch.
Looking at the best mix per image showed that Zero-DCE is weak in the darkest regions,
which gave us the idea to use T directly. Full log: `results/tune_fusion.txt`.

### 3.7 Post-processing
* **Denoising**: bilateral filter in YCrCb, stronger on colour channels (chroma noise
  looks worst). The result is mixed back using T: dark areas, which were amplified most,
  get full denoising; bright areas keep their original detail.
* **CLAHE** on the L channel of LAB only, so colours do not shift. Clip limit is lower
  for noisy images because CLAHE also boosts noise.
* **Colour**: grey-world white balance, but only partially (capped at 0.3) and only when
  the cast is strong. Then a soft knee above 0.85 so lamps fade to white smoothly instead
  of clipping.

## 4. Why these choices

| Choice | Reason | Alternative we did not pick |
|---|---|---|
| Zero-DCE | tiny, runs on CPU, does not need paired data, curves cannot create fake colours | EnlightenGAN / RetinexNet: much bigger, GAN can hallucinate textures |
| Guided filter for T | edge aware, fast, easy to explain | Gaussian blur (halos), optimisation like LIME (slow) |
| Bilateral denoise | asked in the problem, fast | Non-local means (better but slower), BM3D (too slow) |
| Fusion weight from T | simple, explainable, best on validation | Mertens exposure fusion (followed noise), a learned fusion network (needs more data and time) |
| Gradio app | quick to build, works on phone, free hosting on Hugging Face | Flask + custom front end |

## 5. Data

We could not use the LOL dataset inside our build environment (it is hosted on Google
Drive), so we built a reproducible paired set and also collected real dark photos.
The evaluation script supports LOL directly (`--lol`).

| Set | Source | Use |
|---|---|---|
| paired train | CBSD68 images 0–47, darkened 3 ways (144 pairs) | fine-tune |
| paired val | CBSD68 images 48–67 (60 pairs) | pick best epoch |
| paired test | Kodak24 + McMaster, 3 ways (126 pairs) | final scores, never used before |
| real train | 44 DICM photos | fine-tune (no ground truth needed) |
| real val | 10 DICM photos | kept aside |
| real test | 10 DICM + 10 LIME + 6 RetinexNet photos | NIQE, unseen real images |

Darkening (`scripts/make_dataset.py`) works in linear light: scale the light, add a
random colour cast (0.85–1.15 per channel), add shot noise plus read noise, convert back
to sRGB and save as 8-bit. The three conditions are *dark* (2–5% light), *extreme*
(0.4–1.2%) and *non-uniform* (a lamp somewhere, light falls off with distance).
The mean brightness of the dark test images is 0.04–0.10, similar to LOL.

## 6. Training

`scripts/finetune_zerodce.py`: start from the original Zero-DCE weights, Adam, lr 2e-4,
40 epochs, 128 px random crops, batch 8, about 13 s per epoch on 2 CPU cores.
Loss = Zero-DCE's four zero-reference losses (spatial consistency, exposure at 0.5,
colour constancy, curve smoothness) + 20 x L1 to the ground truth for paired images.
Real photos only use the zero-reference losses. The best validation epoch is saved.

## 7. Evaluation

Main result on the unseen test set: hybrid PSNR 17.66 dB / SSIM 0.569, classical
17.09 / 0.562, Zero-DCE fine-tuned 15.90 / 0.383. On real dark photos the hybrid has
0.03% over-exposed pixels (Zero-DCE alone: 1.62%) with NIQE 4.49.

See `eval_P1_TXX.md` for all numbers, `ABLATION_AND_FAILURES.md` for ablation and failure
cases. Metrics: PSNR, SSIM, LPIPS (AlexNet), CIEDE2000 colour error, edge correlation,
remaining noise, over-exposed pixels, NIQE on real photos and latency.

## 8. Deployment

Gradio app (`app.py`) hosted on Hugging Face Spaces (CPU). Images larger than 1280 px are
resized so the answer comes in a few seconds. Error messages for no image or a too small
image. A Dockerfile is included for any other host.

## 9. Limitations

* Our paired data is synthetic. Real camera noise is more complex (it is not just
  Gaussian, and JPEG compression makes blocks). Scores on LOL will be different.
* Grey-world white balance can still change scenes that really are one colour (sunset,
  neon). We capped it, but it is not solved.
* Bilateral filtering smooths fine texture like grass when the noise is very high.
* Zero-DCE has a non-commercial licence, so this project is for research/education.

## 10. Ethics and responsible use

* Enhanced images should not be used as evidence (for example in court or by police)
  without saying they were processed: the denoising and fusion change pixel values and
  can remove or soften small details.
* We do not store uploaded images. The app processes them in memory only.
* Datasets used (CBSD68, Kodak24, McMaster, DICM, LIME) are public research sets.


ewpage

# Evaluation Results – Problem P1 (Low-Light Image Restoration), Team TXX

All numbers are produced by `python scripts/evaluate.py` and `python scripts/ablation.py`.
Raw files are in `results/`. Hardware: 2 CPU cores (Intel Xeon 2.1 GHz), no GPU.

**Test data (never used for training or tuning):**

* Paired: 42 clean photos (Kodak24 + McMaster) darkened 3 ways = 126 pairs
  (`dark`, `extreme`, `nonuniform`, see DESIGN.md section 5).
* Unpaired real dark photos: 26 images (10 DICM, 10 LIME, 6 RetinexNet samples), resized to max 800 px.

Methods compared: input (nothing done), classical only, deep learning only (Zero-DCE
original weights and our fine-tuned weights), hybrid (ours).

## 1. PSNR and SSIM (paired test set, 126 images)

| Method | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|
| Input (no enhancement) | 8.82 | 0.241 |
| Classical only | 17.09 | 0.562 |
| Deep learning only – Zero-DCE pretrained | 15.16 | 0.375 |
| Deep learning only – Zero-DCE fine-tuned | 15.90 | 0.383 |
| **Hybrid (ours)** | **17.66** | **0.569** |

Per lighting condition (generalisation across illumination):

| Condition | Classical PSNR / SSIM | DL fine-tuned PSNR / SSIM | Hybrid PSNR / SSIM |
|---|---|---|---|
| dark | 17.94 / 0.616 | 18.52 / 0.462 | **19.08 / 0.628** |
| extreme | 16.11 / 0.459 | 13.36 / 0.221 | **17.10 / 0.465** |
| non-uniform | **17.22** / 0.612 | 15.81 / 0.466 | 16.81 / **0.613** |

Fine-tuning Zero-DCE raised validation PSNR from 16.28 to 17.68 dB (best epoch 35 of 40)
and test PSNR from 15.16 to 15.90 dB.

## 2. LPIPS (perceptual quality, lower is better)

| Method | LPIPS ↓ |
|---|---|
| Classical only | to be filled |
| Deep learning only (fine-tuned) | to be filled |
| Hybrid (ours) | to be filled |

LPIPS needs the AlexNet weights from download.pytorch.org, which our build machine could
not reach. The code computes it automatically (`src/metrics.py`, `lpips_score`) on any
machine with internet: run `python scripts/evaluate.py --skip_unpaired` and copy the
LPIPS column of `results/paired_synth_summary.csv` here.

## 3. NIQE (no-reference quality, real dark photos, lower is better)

| Method | NIQE ↓ | Mean brightness | Over-exposed pixels (%) ↓ |
|---|---|---|---|
| Input (no enhancement) | 5.11 | 0.196 | 1.57 |
| Classical only | 4.55 | 0.370 | 0.47 |
| DL only – Zero-DCE pretrained | 4.75 | 0.381 | 1.86 |
| DL only – Zero-DCE fine-tuned | **4.41** | 0.422 | 1.62 |
| **Hybrid (ours)** | 4.49 | 0.410 | **0.03** |

Per source, hybrid NIQE: DICM 4.02, LIME 4.72, RetinexNet 4.90.
The DL branch alone has slightly better NIQE, but it clips 1.6% of pixels to white;
the hybrid keeps almost no clipped pixels with only 0.08 worse NIQE.

## 4. Processing latency (CPU, 2 cores)

| Method | Paired test, ~768x512 (ms) | Real photos, max 800 px (ms) |
|---|---|---|
| Classical only | 77 | 98 |
| DL only (Zero-DCE, 512 px curves) | 361 | 229 |
| Hybrid (ours) | 449 | 333 |

In the deployed app a full request (upload, processing, NIQE, response) took about 1.6 s.
`torch.set_flush_denormal(True)` cut Zero-DCE time on one 512 px image from 13.6 s to 0.37 s.

## 5. Noise suppression and edge preservation (paired test set)

| Method | Remaining noise σ ↓ | Edge correlation with ground truth ↑ | Colour error ΔE2000 ↓ |
|---|---|---|---|
| Input (no enhancement) | 0.017 | 0.748 | 33.33 |
| Classical only | 0.031 | 0.759 | 13.14 |
| DL only – pretrained | 0.068 | 0.622 | 19.62 |
| DL only – fine-tuned | 0.074 | 0.656 | 18.81 |
| **Hybrid (ours)** | **0.028** | **0.763** | **12.68** |

Noise σ is the wavelet estimate (`skimage.restoration.estimate_sigma`) on the output
(the dark input has low σ only because its values are tiny). Edge correlation = Pearson
correlation of Sobel gradient magnitude between output and ground truth.

## 6. Ablation summary (full table in ABLATION_AND_FAILURES.md)

| Removed part | PSNR change | SSIM change |
|---|---|---|
| adaptive parameters (fixed values) | −1.34 | −0.063 |
| fusion (DL branch only) | −1.84 | −0.080 |
| fusion replaced by 50/50 average | −0.33 | −0.026 |
| denoising | −2.37 | −0.211 |
| CLAHE | −0.12 | +0.022 |
| fine-tuning (original weights) | −0.22 | +0.007 |
| pre-gain | −0.03 | +0.014 |
| colour correction | +0.13 | +0.002 (but over-exposed pixels rise from 0.0001% to 0.017%) |

## 7. Reproduce

```
python scripts/make_dataset.py
python scripts/finetune_zerodce.py
python scripts/tune_fusion.py
python scripts/evaluate.py
python scripts/ablation.py
python scripts/evaluate.py --lol data/LOL/eval15     # if you have LOL
```

# Ablation Study and Failure Analysis

## 1. Ablation study

We switch off one part at a time and run the same unseen test set
(126 synthetic pairs for PSNR/SSIM/ΔE/edges, 26 real dark photos for NIQE).
Source: `results/ablation.csv`, made by `scripts/ablation.py`.

### Hybrid pipeline

| Variant | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | Edge corr ↑ | Over-exp % ↓ | NIQE real ↓ |
|---|---|---|---|---|---|---|
| **Hybrid full** | **17.66** | 0.569 | 12.68 | 0.763 | 0.0001 | 4.49 |
| without adaptive parameters (fixed values) | 16.31 | 0.506 | 15.30 | 0.703 | 0.0093 | 4.53 |
| without pre-gain | 17.63 | 0.582 | 12.73 | 0.776 | 0.0003 | 4.50 |
| without fusion (Zero-DCE branch only) | 15.82 | 0.488 | 15.43 | 0.677 | 0.0000 | 4.70 |
| fusion replaced by fixed 50/50 average | 17.33 | 0.543 | 13.13 | 0.732 | 0.0000 | 4.62 |
| without denoising | 15.29 | 0.357 | 19.75 | 0.609 | 0.0000 | 4.53 |
| without CLAHE | 17.54 | 0.590 | 13.25 | 0.780 | 0.0000 | 4.61 |
| without colour correction | 17.78 | 0.571 | 12.43 | 0.763 | 0.0169 | 4.46 |
| without fine-tuning (original Zero-DCE weights) | 17.44 | 0.576 | 12.92 | 0.760 | 0.0000 | 4.59 |

### Classical pipeline

| Variant | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | Edge corr ↑ | NIQE real ↓ |
|---|---|---|---|---|---|
| **Classical full** | **17.09** | 0.562 | **13.14** | 0.759 | 4.55 |
| Retinex replaced by global gamma | 16.09 | 0.508 | 15.02 | 0.566 | 4.46 |
| without adaptive parameters | 15.59 | 0.486 | 15.90 | 0.689 | 4.41 |
| without denoising | 14.81 | 0.341 | 20.22 | 0.592 | 4.82 |
| without CLAHE | 15.93 | 0.578 | 15.02 | 0.787 | 4.73 |

### What we learn

1. **Denoising matters most.** Without it SSIM falls from 0.569 to 0.357. Brightening a
   dark photo by 10x also makes the noise 10x larger.
2. **Fusion is the second biggest part.** Zero-DCE alone is 1.84 dB worse. Using the
   illumination map to decide where to trust each branch is 0.33 dB better than a plain
   average, so the adaptive weighting is doing real work, not just averaging.
3. **Adaptive parameters** add 1.34 dB over fixed values. Fixed values over-smooth clean
   images and under-smooth very noisy ones.
4. **Retinex vs global gamma:** edge correlation drops from 0.759 to 0.566 with a global
   gamma, because a single curve cannot brighten shadows without washing out lit areas.
5. **Honest mixed results:**
   * CLAHE raises PSNR and NIQE but lowers SSIM a little (it boosts contrast beyond the
     ground truth). We keep it because real photos look flat without it (NIQE 4.61 → 4.49).
   * Removing colour correction gives +0.13 dB here, because our synthetic colour cast is
     random and grey-world does not always undo it. But without the highlight protection
     inside that step, over-exposed pixels rise 170x, so we keep it.
   * Pre-gain gives almost no change on this test set (−0.03 dB). It was added for
     extremely dark real photos; we keep it but it is a candidate to remove.
   * Fine-tuning helps the DL-only model by 0.74 dB, but inside the hybrid the gain is
     only 0.22 dB, because the Retinex branch already covers Zero-DCE's weak dark areas.

## 2. Failure cases

![failures](docs/failures.jpg)

Columns: input | classical | Zero-DCE | hybrid | ground truth. These are the worst hybrid
result for each lighting condition (`scripts/make_figures.py`).

**Failure 1 – bright white scene stays grey (McMaster_4, dark, PSNR 9.8).**
The ground truth is almost white (white background and curtain). Our brightness target is
0.45, so the output is a medium grey. The system has no way to know the true scene was
that bright; it chooses "average" exposure. Possible fix: predict the target from the
image content (a small regressor), or let the user move a brightness slider.

**Failure 2 – extreme darkness: colour noise and lost texture (kodim20, extreme, PSNR 11.7).**
At 0.4–1.2% light, only a few photon levels are left. After brightening, the grass is
mostly noise, the bilateral filter cannot separate grass texture from noise, and some
coloured speckles remain on the plane. Classical and DL both fail here; the hybrid is
cleaner than Zero-DCE but still noisy. Possible fix: a learned denoiser (e.g. a small
DnCNN) or burst/multi-frame input.

**Failure 3 – non-uniform light: dark corners and colour tint (kodim12, nonuniform, PSNR 13.3).**
The lamp position leaves one corner almost black. The illumination floor (0.02) limits
how much that corner is lifted, so it stays darker than the rest, and the random colour
cast is not fully removed because white balance is capped at 0.3. This is also the one
condition where the classical pipeline beats the hybrid on PSNR (17.22 vs 16.81 dB on
average). Possible fix: a lower floor plus stronger local denoising in very dark regions.

**Failure 4 – real photo, coloured scenes (LIME/7, purple background, see `docs/compare_real.jpg`).**
Zero-DCE turns the purple wall bright magenta and grey-world white balance tries to
"neutralise" it. We capped white balance at 0.3 after seeing a sunset turned blue in our
first test (DICM/05). The hybrid looks acceptable, but any scene that really is one
colour is at risk.

## 3. Limits of this evaluation

* Paired scores come from synthetic darkening, not real camera pairs. Scores on LOL will
  be different; run `python scripts/evaluate.py --lol path/to/LOL/eval15`.
* NIQE was designed for natural daytime images and can prefer brighter, sharper outputs
  even when they are less faithful.
* LPIPS could not be computed on our build machine (no access to the AlexNet weights).

# Appendix: example results

![Real dark photos: input, classical, Zero-DCE, hybrid](docs/compare_real.jpg)

![Synthetic test: input, classical, Zero-DCE, hybrid, ground truth](docs/compare_synthetic.jpg)

![PSNR and SSIM per lighting condition](docs/results_chart.png)

![Intermediate steps of the hybrid pipeline](docs/intermediate.jpg)
