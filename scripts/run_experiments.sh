#!/usr/bin/env bash
set -e

CFG=config.yaml
PYTHON=python3

echo "=== EVENT DETECTION EXPERIMENTS ==="

echo "1) Data prep..."
$PYTHON src/event/data_prep.py

echo "2) Baseline training..."
$PYTHON src/event/baseline_train.py

echo "3) Augment train datasets (x2, x5)..."
$PYTHON src/event/augment.py

echo "4) Train Transformer on original and augmented (multi-seed)"
OUTDIR=experiments/event/outputs/distil_roberta_orig
mkdir -p $OUTDIR
for SEED in 42 7 123 2023 999; do
 echo "seed $SEED original..."
 $PYTHON src/event/transformer_train.py --train data/processed/event/train.csv --val data/processed/event/val.csv --test data/processed/event/test.csv --out $OUTDIR --seed $SEED
done

# augmented x2
OUTDIR2=experiments/event/outputs/distil_roberta_aug_x2
mkdir -p $OUTDIR2
for SEED in 42 7 123 2023 999; do
  echo "seed $SEED aug x2..."
  $PYTHON src/event/transformer_train.py --train data/augmented/event_train_aug_x2.csv --val data/processed/event/val.csv --test data/processed/event/test.csv --out $OUTDIR2 --seed $SEED
done

# augmented x5
OUTDIR5=experiments/event/outputs/distil_roberta_aug_x5
mkdir -p $OUTDIR5
for SEED in 42 7 123 2023 999; do
  echo "seed $SEED aug x5..."
  $PYTHON src/event/transformer_train.py --train data/augmented/event_train_aug_x5.csv --val data/processed/event/val.csv --test data/processed/event/test.csv --out $OUTDIR5 --seed $SEED
done

echo "5) Summarize metrics..."
$PYTHON src/event/evaluate.py experiments/event/outputs/distil_roberta_orig
$PYTHON src/event/evaluate.py experiments/event/outputs/distil_roberta_aug_x2
$PYTHON src/event/evaluate.py experiments/event/outputs/distil_roberta_aug_x5

echo "6) Statistical tests (orig vs aug x2)"
$PYTHON src/event/stats_test.py experiments/event/outputs/distil_roberta_orig experiments/event/outputs/distil_roberta_aug_x2
