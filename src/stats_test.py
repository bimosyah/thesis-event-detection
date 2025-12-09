"""
Statistical significance testing module
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def load_predictions(results_path: str, model_name: str) -> np.ndarray:
    """
    Load predictions from results file

    Args:
        results_path: Path to results JSON file
        model_name: Name of the model

    Returns:
        Array of predictions
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    # This is a placeholder - you'll need to save actual predictions
    # For now, we'll work with the metrics
    return results.get(model_name, {})


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform paired t-test

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level

    Returns:
        Tuple of (t-statistic, p-value, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = p_value < alpha

    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}, significant={is_significant}")

    return t_stat, p_value, is_significant


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform Wilcoxon signed-rank test

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level

    Returns:
        Tuple of (statistic, p-value, is_significant)
    """
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    is_significant = p_value < alpha

    logger.info(f"Wilcoxon test: statistic={statistic:.4f}, p={p_value:.4f}, significant={is_significant}")

    return statistic, p_value, is_significant


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    true_labels: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform McNemar's test

    Args:
        predictions_a: Predictions from model A
        predictions_b: Predictions from model B
        true_labels: True labels
        alpha: Significance level

    Returns:
        Tuple of (statistic, p-value, is_significant)
    """
    # Create contingency table
    correct_a = predictions_a == true_labels
    correct_b = predictions_b == true_labels

    # Both correct or both wrong (ignored in McNemar's test)
    both_correct = np.sum(correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    # Discordant pairs
    a_correct_b_wrong = np.sum(correct_a & ~correct_b)
    b_correct_a_wrong = np.sum(~correct_a & correct_b)

    # McNemar's test statistic
    table = [[both_correct, a_correct_b_wrong],
             [b_correct_a_wrong, both_wrong]]

    # Using chi-square approximation
    statistic = (abs(a_correct_b_wrong - b_correct_a_wrong) - 1)**2 / (a_correct_b_wrong + b_correct_a_wrong)
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    is_significant = p_value < alpha

    logger.info(f"McNemar's test: statistic={statistic:.4f}, p={p_value:.4f}, significant={is_significant}")

    return statistic, p_value, is_significant


def compare_models(
    results_path: str = "experiments/outputs/evaluation_results.json",
    baseline_model: str = "baseline",
    comparison_models: List[str] = None,
    alpha: float = 0.05
) -> Dict:
    """
    Compare models using statistical tests

    Args:
        results_path: Path to evaluation results
        baseline_model: Name of baseline model
        comparison_models: List of models to compare against baseline
        alpha: Significance level

    Returns:
        Dictionary containing test results
    """
    if comparison_models is None:
        comparison_models = [
            'distil_roberta_orig',
            'distil_roberta_aug_x2',
            'distil_roberta_aug_x5'
        ]

    logger.info("Performing statistical significance tests")

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    test_results = {}

    baseline_f1 = results[baseline_model]['f1_score']

    for model_name in comparison_models:
        if model_name not in results:
            logger.warning(f"Model {model_name} not found in results")
            continue

        model_f1 = results[model_name]['f1_score']

        logger.info(f"\nComparing {baseline_model} vs {model_name}")
        logger.info(f"  {baseline_model} F1: {baseline_f1:.4f}")
        logger.info(f"  {model_name} F1: {model_f1:.4f}")
        logger.info(f"  Difference: {model_f1 - baseline_f1:.4f}")

        # For now, store the comparison
        # In practice, you'd need cross-validation scores or bootstrap samples
        test_results[f"{baseline_model}_vs_{model_name}"] = {
            'baseline_f1': baseline_f1,
            'comparison_f1': model_f1,
            'difference': model_f1 - baseline_f1,
            'improvement_percentage': ((model_f1 - baseline_f1) / baseline_f1) * 100
        }

    # Save test results
    output_path = "experiments/outputs/statistical_tests.json"
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=4)

    logger.info(f"\nStatistical test results saved to {output_path}")

    return test_results


if __name__ == "__main__":
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    results = compare_models()

    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*50)
    for comparison, metrics in results.items():
        print(f"\n{comparison}:")
        print(f"  Difference: {metrics['difference']:.4f}")
        print(f"  Improvement: {metrics['improvement_percentage']:.2f}%")

