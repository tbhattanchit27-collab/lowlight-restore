# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | `P1` |
| **Problem Title** | `Robust Low-Light Image Restoration and Enhancement` |
| **Team Name** | `APex decs` |
| **Team Members** | `Anish Gowda P (USN: 1RUA24CSE0053), Anchit Tushar Bhatt (USN: 1RUA24CSE0049), Amith S (USN: 1RUA24CSE0038), Chethan Chouhan (USN: 1RUA24CSE0104)` |
| **GitHub Repository** | https://github.com/tbhattanchit27-collab/lowlight-restore |
| **Live Demo / Deployment** |   |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** Photos taken in the dark have four problems at once:
  they are very underexposed, they are full of sensor noise that gets bigger when the
  image is brightened, they often have a colour cast, and the light is uneven (a lamp in
  one corner, deep shadow in another). Simply raising brightness makes the noise worse and
  turns lamps pure white. Our constraints were to run on a normal laptop CPU with no GPU,
  answer in under about half a second per 600x400 image, and show every step in a web app.
* **Proposed Methodology:** We run a classical Retinex method and a small deep learning
  model (Zero-DCE) side by side, then mix them pixel by pixel using the estimated
  illumination map: the darkest regions take the Retinex result and the lit regions take
  Zero-DCE. Every strength (brightening target, denoising, CLAHE, white balance) is computed
  from measurements of the input image, and the two fusion settings were chosen on a
  validation set. We chose this because neither method alone works everywhere: on the
  LOL-v2 Real test set the hybrid scores 19.63 dB, against 18.02 dB (classical) and
  18.12 dB (Zero-DCE).

---

## 2. Technical Architecture & Implementation

The pipeline diagram is in `docs/architecture.png`. Code: `src/classical.py`,
`src/zerodce.py`, `src/hybrid.py`, metrics in `src/metrics.py`, app in `app.py`.

### 2.1 Classical Algorithmic Baseline
* **Algorithms Utilized:**
  1. **Illumination map T** (LIME idea): the maximum of R, G, B for every pixel, smoothed
     with a **guided filter** that uses the image itself as the guide, so edges between
     bright and dark objects stay sharp and no halos appear.
  2. **Retinex enhancement:** image = reflectance x illumination, so we change only the
     light: output = I x T^(gamma - 1). A floor of 0.02 on T stops black pixels exploding.
  3. **Bilateral filter** in YCrCb, stronger on the two colour channels, because colour
     noise looks worst.
  4. **CLAHE** on the L channel of LAB only, so colours do not shift.
  5. **Grey-world white balance** (partial) and a soft knee above 0.85 so bright lights
     fade to white instead of clipping.
* **Adaptive Selection Logic:** For each input we measure mean brightness, share of very
  dark pixels, noise level (wavelet estimate), colour cast, contrast (95th minus 5th
  percentile) and how uneven the light is. From these:
  * gamma is solved so the mean of the new illumination reaches 0.45 (0.38 for extremely
    dark images, so we do not just reveal noise);
  * denoise strength = noise level x expected brightening gain x 6, clipped to 0..1, so a
    clean photo is barely smoothed and a dark noisy one is smoothed strongly. Denoising is
    also blended by T, so the darkest regions (amplified most) get the most smoothing;
  * the CLAHE clip limit is 2.0 for flat images and 1.5 otherwise, reduced further when
    the image is noisy;
  * white balance is applied only when the colour cast is strong, and at most 30%.

### 2.2 Deep Learning Pipeline
* **Model Architecture:** **Zero-DCE** (Guo et al., CVPR 2020), a 7-layer CNN with
  79,416 parameters. Instead of an image it predicts 8 per-pixel curves,
  x + r(x² − x), which are smooth and always stay in 0..1, so it cannot invent colours or
  textures. We chose it over EnlightenGAN or RetinexNet because it is tiny enough for a
  CPU web app and a GAN can hallucinate detail. Our changes: curves are predicted on a copy
  of at most 512 px and upsampled; the output after 4 curves and after 8 curves are both
  computed and the one with brightness closer to 0.5 and fewer clipped pixels is kept;
  `torch.set_flush_denormal(True)` cut CPU time for one 512 px image from 13.6 s to 0.37 s.
* **Loss Functions & Training:** We started from the authors' weights and fine-tuned on
  the official **LOL-v2** dataset (Real_captured + Synthetic Train, 1,589 pairs: 1,431 for
  training, 158 held back for validation, random seed 0). Loss = the four Zero-DCE losses
  (spatial consistency, exposure control at 0.5, colour constancy, smoothness of the curve
  maps) + 20 x L1 distance to the ground truth. Adam, learning rate 2e-4, 15 epochs,
  128 px crops, batch 8, on a MacBook CPU. Validation PSNR rose from 13.23 to 14.58 dB
  (best epoch 6).

### 2.3 Hybrid Integration & Adaptive Mechanism
* **Fusion Strategy:** Classical priors guide the network in two places.
  1. **Pre-gain:** when the image mean brightness is below 0.10, it is first lifted with the
     Retinex map to a mean of about 0.20 before going into Zero-DCE, because Zero-DCE was
     trained on images that are dark but not black.
  2. **Illumination-guided fusion:** each pixel gets a Zero-DCE weight
     w = clip(T / 0.2, 0, 1), smoothed with a Gaussian (sigma 5) to avoid seams, and
     output = w x ZeroDCE + (1 − w) x Retinex.
  The threshold 0.2 and the CLAHE strength 0.35 were chosen on the 158 validation pairs
  (`results/tune_fusion_official.txt`). Our first fusion rule (exposure-fusion weights)
  did worse than a plain average, because Zero-DCE's noise looked like "detail" and
  attracted weight.
* **Decision Logic:** Zero-DCE gives natural brightness and colour where there is some
  light, but in the darkest areas it under-lights and amplifies noise. The Retinex branch
  is steadier there. So the illumination map decides: low T means trust Retinex and denoise
  hardest; high T means trust Zero-DCE and keep detail. After fusion, denoising strength
  follows the noise estimate and is spatially weighted by T, CLAHE is applied lightly, and
  the highlight knee protects bright regions, giving 0% over-exposed pixels on all three
  test sets.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

Official **LOL-v2 Real_captured Test** set (100 pairs, never used for training or tuning).
LPIPS on the same 100 pairs; NIQE on 26 real dark photos without ground truth (DICM, LIME,
RetinexNet samples). Latency on a MacBook Pro CPU, 600x400 images.

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric (LPIPS / NIQE) ↓ | Latency (ms / frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Baseline** | `18.02` | `0.673` | `0.286 / 4.55` | `69 ms` |
| **Deep Learning Only** | `18.12` | `0.580` | `0.313 / 4.75` | `380 ms` |
| **Proposed Hybrid System** | `19.63` | `0.761` | `0.230 / 4.76` | `447 ms` |

Deep Learning Only = Zero-DCE with the authors' weights (the better of the two Zero-DCE
variants on this set; our fine-tuned version scored 15.58 dB / 0.485). Other test sets:

| Test set | Classical PSNR / SSIM | Zero-DCE (best) PSNR / SSIM | Hybrid PSNR / SSIM |
| :--- | :---: | :---: | :---: |
| LOL-v2 Synthetic Test (100) | 15.51 / 0.671 | 16.98 / 0.724 | 16.95 / 0.722 |
| LOL v1 eval15 (15) | 16.37 / 0.656 | 16.79 / 0.520 | 18.37 / 0.742 |

* **Analysis:** The hybrid gives the best structural fidelity (+1.5 dB PSNR, +0.09 SSIM
  over the best single method) and the best LPIPS on real captured data, at about 6.5x
  the classical method's time, since Zero-DCE takes most of the 447 ms. NIQE slightly
  prefers the classical output (4.55 vs 4.76) because NIQE rewards strong local contrast,
  while the hybrid is smoother and closer to the reference photos. On LOL-v2 Synthetic,
  which has little real noise, the hybrid only ties with Zero-DCE.

### 3.2 Visual Outputs & Intermediate Representations
All figures are made by `python scripts/make_figures.py` and stored in `docs/`.

1. **Extreme low-light:** `docs/compare_lolv2.jpg`, rows 1–2: the two darkest LOL-v2 Real
   test images, shown as input | classical | Zero-DCE | hybrid | ground truth.
2. **Non-uniform / backlit scenes:** `docs/compare_lolv2.jpg`, row 3 (the LOL-v2 test
   image with the most uneven light) and `docs/compare_real.jpg` (real night photos with
   lamps and windows). Bright regions are kept below clipping by the highlight knee:
   0% over-exposed pixels for the hybrid against 0.013% for Zero-DCE on LOL-v2 Real.
3. **Intermediate states:** `docs/intermediate.jpg` shows input, illumination map T,
   Retinex branch, Zero-DCE branch, fusion weight map (bright = Zero-DCE), fused, denoised
   and final output. The web app shows the same steps for any uploaded photo, together with
   the measured noise level and the parameters chosen from it.

### 3.3 Ablation Study

Each part is switched off once and the same 100 LOL-v2 Real test pairs are re-run
(`results/ablation_lolv2_real.csv`).

| Change to the hybrid | PSNR (dB) | Δ PSNR | SSIM | Δ SSIM |
| :--- | :---: | :---: | :---: | :---: |
| Full hybrid | 19.63 | – | 0.761 | – |
| No classical guidance in fusion (Zero-DCE branch only) | 13.91 | −5.73 | 0.618 | −0.144 |
| Fusion replaced by a fixed 50/50 average | 18.01 | −1.62 | 0.715 | −0.046 |
| No denoising | 18.04 | −1.59 | 0.497 | −0.265 |
| Fixed instead of adaptive parameters | 19.33 | −0.30 | 0.741 | −0.020 |
| No pre-gain | 19.36 | −0.27 | 0.759 | −0.002 |
| No colour correction | 19.45 | −0.18 | 0.760 | −0.001 |
| Authors' Zero-DCE weights instead of fine-tuned | 19.67 | +0.03 | 0.762 | +0.001 |
| No CLAHE | 20.29 | +0.66 | 0.784 | +0.023 |

Classical guidance is the single most important part, and illumination-based weighting
beats plain averaging by 1.62 dB. Adaptive parameters help and also stop clipping (fixed
parameters give 0.006% over-exposed pixels). Removing CLAHE would score higher on the test
set, but we did not choose settings from the test set: on the validation set the light
CLAHE setting (0.35) scored best (14.72 dB, against 14.44 dB with CLAHE off), so we kept it.

---

## 4. Failure Modes & Limitations

* **Observed Failure Case:** Fine-tuning made Zero-DCE alone *worse* on real captured
  photos: on LOL-v2 Real it fell from 18.12 to 15.58 dB and its remaining noise rose from
  0.032 to 0.056, while it improved on LOL-v2 Synthetic (16.69 → 16.98 dB) and eval15
  (14.98 → 16.79 dB). The three test images where the hybrid scores lowest are shown in
  `docs/failures_lolv2.jpg` (input | classical | Zero-DCE | hybrid | ground truth).
* **Root Cause & Mitigation:** Training and validation mixed 689 Real and 900 Synthetic
  pairs, so the validation score was mostly synthetic and the saved epoch was the one that
  brightens aggressively. Real sensor noise is much heavier, and a curve that multiplies
  brightness multiplies noise by the same factor, with no step in Zero-DCE to remove it.
  In the hybrid this is mostly hidden because the darkest regions use the Retinex branch
  and are denoised hardest (the hybrid is equal with either weights: 19.63 vs 19.67 dB).
  Fixes: validate on Real images only, or keep one model per domain; and replace the
  bilateral filter with a small learned denoiser for extreme shadows, where bilateral
  filtering cannot separate fine texture from noise. Other limits: very bright scenes
  come out too grey (we aim at average brightness), and scenes that are truly one colour
  can be shifted by the grey-world white balance.

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands
```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/tbhattanchit27-collab/lowlight-restore.git
cd lowlight-restore
pip install -r requirements.txt

# 2. Run inference on a sample directory
python run_inference.py --input_dir ./samples/ --output_dir ./results/

# optional: classical or DL only, and save the intermediate maps
python run_inference.py --input_dir ./samples/ --output_dir ./out/ --method classical
python run_inference.py --input_dir ./samples/ --output_dir ./out/ --save_steps

# 3. Web app (http://localhost:7860)
python app.py

# 4. Reproduce all benchmark numbers (LOL-v2 in ./LOL-v2, LOL v1 in ./LOLdataset)
sh scripts/run_official.sh LOL-v2 LOLdataset
```

### Deployment Format
* **Gradio UI** (`app.py`): upload a photo, choose Hybrid / Classical / DL, see a
  before/after slider, every intermediate step and the measured image statistics, plus a
  tab that runs all three methods side by side with timings. Large photos are resized to
  1280 px, and there are clear error messages for a missing or too-small image.
* **Docker container** (`Dockerfile`): `docker build -t lowlight . && docker run -p 7860:7860 lowlight`.
* **CLI script** (`run_inference.py`) for folders of images, which also writes `latency.csv`.
* Hosted on Hugging Face Spaces (CPU); see `docs/DEPLOY.md`.

---

### Student Submission Checklist
- [x] Working GitHub repository link included with an updated `README.md`.
- [x] No raw code pasted into this report.
- [x] All table metrics match test runs reproducible via the repository.
- [x] All team member details accurately recorded.
