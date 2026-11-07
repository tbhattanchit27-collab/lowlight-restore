# Viva notes (for the team, not for judges)

## Suggested split of parts
| Member | Owns | Files |
|---|---|---|
| 1 | classical pipeline: analysis, illumination, Retinex, bilateral, CLAHE, colour | src/classical.py |
| 2 | Zero-DCE, fine-tuning, losses | src/zerodce.py, scripts/finetune_zerodce.py |
| 3 | hybrid fusion, tuning, ablation | src/hybrid.py, scripts/tune_fusion.py, scripts/ablation.py |
| 4 | dataset, metrics, app, deployment | scripts/make_dataset.py, src/metrics.py, app.py |

## 3-minute demo script
1. (20 s) Problem: dark photos are dark, noisy, colour shifted. Brightening alone makes noise worse.
2. (40 s) Upload a judge's photo -> Hybrid. Drag the before/after slider.
3. (50 s) Show "Intermediate steps": illumination map, the two branches, the fusion
   weight map ("dark areas use Retinex, lit areas use Zero-DCE"), denoised, final.
   Point at the parameter table: "noise sigma was X, so denoise strength became Y".
4. (30 s) "Compare all three" tab: classical, DL, hybrid with time and NIQE.
5. (40 s) Numbers on the official LOL-v2 Real test (100 pairs): hybrid 19.63 dB / SSIM
   0.761 / LPIPS 0.230, classical 18.02 / 0.673, Zero-DCE 18.12 / 0.580. Then one honest
   failure: fine-tuning on mixed Real + Synthetic data hurt Zero-DCE on Real photos.

## Questions you should be able to answer

**What is Retinex?** Image = reflectance x illumination. We estimate illumination T and
change it (T to T^gamma), keeping reflectance, so colours stay the same.

**How do you get T?** max(R,G,B) per pixel, then a guided filter with the grey image as
guide. Guided filter = local linear model `q = a*I + b` in each window; it smooths flat
areas but keeps edges, so no halos.

**How is gamma chosen?** So that mean(T^gamma) = target 0.45: gamma = log(0.45) / log(mean T).

**What is CLAHE?** Histogram equalisation in small tiles (8x8 grid), with the histogram
clipped (clip limit) so noise is not boosted too much, and tiles blended bilinearly.
We apply it only on L of LAB so colours do not shift.

**Bilateral filter?** Weighted average where weights depend on distance AND intensity
difference, so pixels across an edge are not averaged. We use stronger settings on
Cr/Cb because colour noise looks worst.

**What does Zero-DCE output?** 24 channels = 8 curves x RGB. Each curve x + r(x^2 - x)
with r in [-1, 1]. 79,416 parameters. "Zero-reference" = trained without ground truth,
using 4 losses: spatial consistency, exposure, colour constancy, smoothness of curves.

**What did you fine-tune?** All Zero-DCE weights, 15 epochs, lr 2e-4, on 1,431 LOL-v2
training pairs (Real + Synthetic), with the four zero-reference losses + 20 x L1 to the
ground truth. Validation PSNR 13.23 -> 14.58 dB (best epoch 6).

**Did fine-tuning help?** Mixed, and we say so. Zero-DCE alone got better on Synthetic
(+0.29 dB) and eval15 (+1.82 dB) but worse on Real (-2.54 dB, noisier). Our validation set
was mostly synthetic images, so the saved epoch suited them. Inside the hybrid it makes no
difference (19.63 vs 19.67 dB), because the Retinex branch handles the dark, noisy parts.
Fix: validate on Real images only.

**Why is your train/test split fair?** We used the official LOL-v2 split. 10% of the
training pairs (158) were held back as validation (seed 0, saved in
data/official_val.json). The epoch, t_high and CLAHE strength were all chosen on
validation. The 100 Real and 100 Synthetic test pairs were only used at the end.

**Why didn't you remove CLAHE when the ablation showed +0.66 dB without it?** That number
comes from the test set. Choosing settings from the test set is overfitting to it. We
added CLAHE strength to the validation tuning instead, and validation chose a light setting
(0.35), not "off".

**Why is eval15 not your main result?** LOL-v2 Real was built partly from LOL v1 photos, so
some eval15 scenes may be in our training data. LOL-v2 Real Test is the clean one.

**Why did a global gamma beat Retinex in the classical ablation?** LOL reference photos
look like a simple global brightening, and Retinex brightens shadows more than lit parts.
Gamma wins PSNR (18.86 vs 18.02) but keeps edges worse (0.827 vs 0.852).

**Why not MIT-Adobe FiveK?** It has no dark/bright pairs (it is expert retouches of normal
photos) and is about 50 GB.

**What is adaptive in your system?** Brightening target, gamma, denoise strength (noise
sigma x gain), CLAHE clip, white balance strength, pre-gain on/off, per-pixel fusion
weight, and the 4-vs-8 curve choice. Ablation on LOL-v2 Real: fixed values cost 0.30 dB
and 0.020 SSIM, and let some pixels clip.

**What is "image-quality-aware"?** Zero-DCE gives an output after 4 and after 8 curves;
we keep the one with brightness closest to 0.5 and fewest clipped pixels. Also the app
reports NIQE and over-exposure for every result.

**Why does the hybrid beat both?** Zero-DCE is weak in the darkest regions (noisy, under-
lit) but natural in lit regions; Retinex is the opposite. Weighting by T uses each where
it is strong. On LOL-v2 Real a plain 50/50 average is 1.62 dB worse, and Zero-DCE
alone is 5.73 dB worse.

**PSNR?** 10 log10(1 / MSE) for images in 0..1. **SSIM?** compares local mean, contrast
and structure, 1 = identical. **LPIPS?** distance between deep features of AlexNet, lower
= perceptually closer. **NIQE?** no-reference: distance of the image's natural-scene
statistics from a model fitted on clean photos; lower = more natural.

**What was the synthetic data for?** Before we had LOL-v2 we built our own pairs by
darkening clean photos in linear light with shot + read noise. We used them to design the
pipeline (fusion rule, white-balance cap). Final numbers are on official LOL-v2.

**Biggest weakness?** Fine-tuning hurt Zero-DCE on Real photos; on LOL-v2 Synthetic the
hybrid is no better than Zero-DCE; NIQE slightly prefers the classical output. Visually,
bright white scenes stay grey, extreme darkness leaves colour noise, and one-colour scenes
can be shifted by white balance.

**Why was the first fusion worse?** Exposure-fusion weights use local contrast, and the
noisy Zero-DCE output has "fake" contrast from noise, so it got too much weight.

**Speed trick?** torch.set_flush_denormal(True): 13.6 s -> 0.37 s for one 512 px image on CPU.
Also curves are predicted at 512 px and upsampled.
