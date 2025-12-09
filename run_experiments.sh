#!/bin/bash

# Event Detection Experiments Runner
# This script runs the complete pipeline for event detection experiments

set -e  # Exit on error

echo "=========================================="
echo "Event Detection Experiments Pipeline"
echo "=========================================="

# Step 1: Data Preparation
echo -e "\n[1/6] Preparing data..."
python src/data_prep.py

# Step 2: Data Augmentation
echo -e "\n[2/6] Augmenting training data..."
python src/augment.py

# Step 3: Train Baseline Model
echo -e "\n[3/6] Training baseline model..."
python src/baseline_train.py

# Step 4: Train Transformer Models
echo -e "\n[4/6] Training transformer models..."
echo "  - Training on original data..."
python src/transformer_train.py

echo "  - Training on 2x augmented data..."
# (Handled in transformer_train.py)

echo "  - Training on 5x augmented data..."
# (Handled in transformer_train.py)

# Step 5: Evaluate All Models
echo -e "\n[5/6] Evaluating all models..."
python src/evaluate.py

# Step 6: Statistical Significance Tests
echo -e "\n[6/6] Running statistical tests..."
python src/stats_test.py

echo -e "\n=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to: experiments/outputs/"
echo "Logs saved to: experiments/logs/"
echo ""
echo "To view results:"
echo "  - Evaluation metrics: experiments/outputs/evaluation_results.json"
echo "  - Statistical tests: experiments/outputs/statistical_tests.json"

