1RUA24CSE0049, 1RUA24CSE0049, 1RUA24CSE0038, 1RUA24CSE0104

# Evaluation – Problem P1 (Low-Light Image Restoration), Team APex decs

All numbers come from `scripts/evaluate.py` and `scripts/ablation.py`, run on a MacBook Pro
(CPU only). Raw files are in `results/` (`paired_lolv2_real_summary.csv` and others).

## Data and split

| Set | Images | Used for |
|---|---|---|
| LOL-v2 Real_captured Train + Synthetic Train | 1,589 pairs | 90% (1,431) fine-tuning, 10% (158) validation |
| **LOL-v2 Real_captured Test** | 100 pairs | **main test result** |
| LOL-v2 Synthetic Test | 100 pairs | test |
| LOL v1 eval15 | 15 pairs | extra test (see note) |
| Real dark photos without ground truth (DICM, LIME, RetinexNet samples) | 26 | NIQE only |

The test sets were never used for training or for choosing any setting. The fusion
setting (t_high = 0.2) and CLAHE strength (0.35) were chosen on the 158 validation pairs
(`results/tune_fusion_official.txt`).

Note on eval15: LOL-v2 Real_captured was built partly from LOL v1 photos, so some eval15
scenes may also appear in LOL-v2's training set. We treat LOL-v2 Real Test as the main,
clean result.

Methods: input (nothing done), classical only, deep learning only (Zero-DCE with original
weights and with our LOL-v2 fine-tuned weights), hybrid (ours, uses the fine-tuned weights).

## 1. PSNR and SSIM

| Method | LOL-v2 Real PSNR / SSIM | LOL-v2 Synthetic PSNR / SSIM | LOL v1 eval15 PSNR / SSIM |
|---|---|---|---|
| Input (no enhancement) | 9.72 / 0.196 | 10.90 / 0.407 | 7.77 / 0.191 |
| Classical only | 18.02 / 0.673 | 15.51 / 0.671 | 16.37 / 0.656 |
| DL only – Zero-DCE pretrained | 18.12 / 0.580 | 16.69 / 0.721 | 14.98 / 0.564 |
| DL only – Zero-DCE fine-tuned | 15.58 / 0.485 | **16.98 / 0.724** | 16.79 / 0.520 |
| **Hybrid (ours)** | **19.63 / 0.761** | 16.95 / 0.722 | **18.37 / 0.742** |

On LOL-v2 Real the hybrid is 1.61 dB above classical and 1.52 dB above the best
Zero-DCE. On LOL-v2 Synthetic the hybrid and fine-tuned Zero-DCE are equal (0.03 dB apart).

## 2. LPIPS (perceptual distance, lower is better)

| Method | LOL-v2 Real | LOL-v2 Synthetic | LOL v1 eval15 |
|---|---|---|---|
| Input (no enhancement) | 0.520 | 0.440 | 0.560 |
| Classical only | 0.286 | 0.321 | 0.291 |
| DL only – Zero-DCE pretrained | 0.313 | **0.254** | 0.335 |
| DL only – Zero-DCE fine-tuned | 0.460 | 0.267 | 0.419 |
| **Hybrid (ours)** | **0.230** | 0.280 | **0.223** |

LPIPS uses AlexNet features (`lpips` package).

## 3. NIQE (no reference, 26 real dark photos, lower is better)

| Method | NIQE |
|---|---|
| Input (no enhancement) | 5.11 |
| Classical only | **4.55** |
| **Hybrid (ours)** | 4.76 |

NIQE favours the classical output slightly. NIQE was designed for daylight photos and
rewards strong local contrast, so it does not always agree with PSNR/SSIM. In the earlier
run with our first weights, Zero-DCE alone scored 4.41 but clipped 1.6% of pixels to white.

## 4. Processing latency (MacBook CPU, LOL images 600x400)

| Method | ms per image |
|---|---|
| Classical only | 69 |
| DL only (Zero-DCE) | 366–380 |
| Hybrid (ours) | 447 |

These are from our second run. In the first run on the same laptop, the times were about
half (classical 34 ms, Zero-DCE 200 ms, hybrid 240 ms), so they depend on what else the
laptop is doing. Zero-DCE takes most of the hybrid's time.

## 5. Noise suppression, edges, colour, exposure (LOL-v2 Real)

| Method | Remaining noise σ ↓ | Edge correlation ↑ | Colour error ΔE2000 ↓ | Over-exposed pixels % ↓ |
|---|---|---|---|---|
| Input (no enhancement) | 0.0067 | 0.818 | 27.24 | 0.011 |
| Classical only | 0.0194 | 0.852 | 13.45 | 0.000 |
| DL only – pretrained | 0.0318 | **0.881** | 14.15 | 0.013 |
| DL only – fine-tuned | 0.0557 | 0.785 | 18.78 | 0.014 |
| **Hybrid (ours)** | **0.0146** | 0.865 | **12.99** | **0.000** |

Noise σ is the wavelet estimate on the output (the dark input has a low σ only because its
pixel values are tiny). Edge correlation = correlation of Sobel gradients with the ground
truth.

## 6. Ablation summary (LOL-v2 Real test, full table in ABLATION_AND_FAILURES.md)

| Change | PSNR change | SSIM change |
|---|---|---|
| fusion removed (Zero-DCE branch only) | −5.73 | −0.144 |
| fusion replaced by fixed 50/50 average | −1.62 | −0.046 |
| denoising removed | −1.59 | −0.265 |
| adaptive parameters replaced by fixed values | −0.30 | −0.020 |
| pre-gain removed | −0.27 | −0.002 |
| colour correction removed | −0.18 | −0.001 |
| original Zero-DCE weights instead of fine-tuned | +0.03 | +0.001 |
| CLAHE removed | +0.66 | +0.023 |

## 7. Fine-tuning

Zero-DCE was fine-tuned for 15 epochs on 1,431 LOL-v2 training pairs. Validation PSNR went
from 13.23 to 14.58 dB (best epoch 6). On the test sets the effect was mixed: better on
Synthetic (+0.29 dB) and eval15 (+1.82 dB), worse on Real (−2.54 dB). Inside the hybrid
it makes no real difference (19.63 vs 19.67 dB). See ABLATION_AND_FAILURES.md for why.

## 8. Reproduce

```
sh scripts/run_official.sh LOL-v2 LOLdataset    # fine-tune, tune, evaluate, ablation
```
or step by step:
```
python scripts/finetune_zerodce.py --train_dir LOL-v2/Real_captured/Train --train_dir LOL-v2/Synthetic/Train --epochs 15
python scripts/tune_fusion.py --official
python scripts/evaluate.py --test_dir LOL-v2/Real_captured/Test --name lolv2_real
python scripts/evaluate.py --test_dir LOL-v2/Synthetic/Test --name lolv2_synthetic
python scripts/evaluate.py --test_dir LOLdataset/eval15 --name lol_v1_eval15
python scripts/ablation.py --test_dir LOL-v2/Real_captured/Test --name lolv2_real
```
