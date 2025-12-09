# Event Detection with Data Augmentation

This repository contains a complete machine learning pipeline for event detection using both baseline models and transformer-based models, with experiments on data augmentation techniques.

## Features

- **Data Preprocessing**: Automated data cleaning, splitting, and preparation
- **Data Augmentation**: Multiple text augmentation techniques (synonym replacement, back translation, etc.)
- **Baseline Models**: Traditional ML models (Naive Bayes, Logistic Regression, SVM)
- **Transformer Models**: State-of-the-art DistilRoBERTa implementation
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Statistical Testing**: Paired t-tests, Wilcoxon tests for model comparison
- **Reproducibility**: All experiments are configurable via YAML and use random seeds

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your dataset to data/raw/dataset.csv

# Run complete pipeline
./run_experiments.sh
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed information about the folder structure and modules.

## Experiments

The pipeline runs the following experiments:
1. **Baseline**: Traditional ML model on original data
2. **Transformer (Original)**: DistilRoBERTa on original training data
3. **Transformer (2x Aug)**: DistilRoBERTa on 2x augmented training data
4. **Transformer (5x Aug)**: DistilRoBERTa on 5x augmented training data

## Configuration

All experiments are configured through `config.yaml`:
- Dataset paths and split ratios
- Augmentation parameters
- Model hyperparameters
- Training settings

## Results

Results are saved to:
- `experiments/outputs/evaluation_results.json` - Performance metrics
- `experiments/outputs/statistical_tests.json` - Statistical comparisons
- `experiments/logs/` - Training logs

## License

MIT

## Author

[Your Name]

