#!/bin/sh
# Train and evaluate on the official LOL-v2 dataset (and LOL v1 eval15 if present).
# usage:  sh scripts/run_official.sh path/to/LOL-v2  [path/to/LOL]
# Takes about 30-45 minutes on a laptop CPU.
set -e
cd "$(dirname "$0")/.."
V2="$1"; V1="$2"
if [ -z "$V2" ] || [ ! -d "$V2" ]; then echo "give the LOL-v2 folder: sh scripts/run_official.sh data/LOL-v2"; exit 1; fi

find_dir () { find "$V2" -type d -path "*$1" | head -1; }
REAL_TRAIN=$(find_dir "Real_captured/Train"); REAL_TEST=$(find_dir "Real_captured/Test")
SYN_TRAIN=$(find_dir "Synthetic/Train");      SYN_TEST=$(find_dir "Synthetic/Test")
echo "Real train: $REAL_TRAIN"; echo "Real test : $REAL_TEST"
echo "Syn  train: $SYN_TRAIN";  echo "Syn  test : $SYN_TEST"
[ -n "$REAL_TRAIN" ] && [ -n "$REAL_TEST" ] || { echo "could not find Real_captured/Train and Test inside $V2"; exit 1; }

TRAIN="--train_dir $REAL_TRAIN"
[ -n "$SYN_TRAIN" ] && TRAIN="$TRAIN --train_dir $SYN_TRAIN"

echo "== 1/4 fine-tune Zero-DCE on the official training images (10% held back for validation)"
python scripts/finetune_zerodce.py $TRAIN --epochs 15

echo "== 2/4 choose the fusion setting on the validation part"
python scripts/tune_fusion.py --official

echo "== 3/4 evaluate on the official test sets"
python scripts/evaluate.py --test_dir "$REAL_TEST" --name lolv2_real
[ -n "$SYN_TEST" ] && python scripts/evaluate.py --test_dir "$SYN_TEST" --name lolv2_synthetic
if [ -n "$V1" ]; then
  EVAL15=$(find "$V1" -type d -name "eval15" | head -1)
  [ -n "$EVAL15" ] && python scripts/evaluate.py --test_dir "$EVAL15" --name lol_v1_eval15
fi

echo "== 4/4 ablation on LOL-v2 real test"
python scripts/ablation.py --test_dir "$REAL_TEST" --name lolv2_real

echo; echo "DONE. Send these files to Claude:"
ls results/paired_lol*_summary.csv results/ablation_lolv2_real.csv results/tune_fusion_official.txt results/finetune_log_zerodce_finetuned.json
