#!/usr/bin/env bash
set -e

CFG=config.yaml
PYTHON=python3

echo "1) Data prep..."
$PYTHON src/data_prep.py

echo "2) Baseline training..."
$PYTHON src/baseline_train.py

echo "3) Augment train datasets (x2, x5)..."
$PYTHON src/augment.py
#
#echo "4) Train Transformer on original and augmented (multi-seed)"
#OUTDIR=experiments/outputs/distil_roberta_orig
#mkdir -p $OUTDIR
#for SEED in 42 7 123 2023 999; do
#  echo "seed $SEED original..."
#  $PYTHON src/transformer_train.py --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv --out $OUTDIR --seed $SEED
#done
#
## augmented x2
#OUTDIR2=experiments/outputs/distil_roberta_aug_x2
#mkdir -p $OUTDIR2
#for SEED in 42 7 123 2023 999; do
#  echo "seed $SEED aug x2..."
#  $PYTHON src/transformer_train.py --train data/augmented/train_aug_x2.csv --val data/processed/val.csv --test data/processed/test.csv --out $OUTDIR2 --seed $SEED
#done
#
## augmented x5
#OUTDIR5=experiments/outputs/distil_roberta_aug_x5
#mkdir -p $OUTDIR5
#for SEED in 42 7 123 2023 999; do
#  echo "seed $SEED aug x5..."
#  $PYTHON src/transformer_train.py --train data/augmented/train_aug_x5.csv --val data/processed/val.csv --test data/processed/test.csv --out $OUTDIR5 --seed $SEED
#done
#
#echo "5) Summarize metrics..."
#$PYTHON src/evaluate.py experiments/outputs/distil_roberta_orig
#$PYTHON src/evaluate.py experiments/outputs/distil_roberta_aug_x2
#$PYTHON src/evaluate.py experiments/outputs/distil_roberta_aug_x5
#
#echo "6) Statistical tests (orig vs aug x2)"
#$PYTHON src/stats_test.py experiments/outputs/distil_roberta_orig experiments/outputs/distil_roberta_aug_x2
