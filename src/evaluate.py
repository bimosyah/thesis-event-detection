"""
Model evaluation module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import logging
from typing import Dict, Any, Tuple
import json

logger = logging.getLogger(__name__)


def evaluate_baseline(
    model_dir: str,
    test_df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Dict[str, Any]:
    """
    Evaluate baseline model

    Args:
        model_dir: Directory containing saved baseline model
        test_df: Test dataframe
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating baseline model from {model_dir}")

    # Load model and vectorizer
    model = joblib.load(f"{model_dir}/model.pkl")
    vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")

    # Vectorize test data
    X_test = vectorizer.transform(test_df[text_column])
    y_test = test_df[label_column]

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(f"Baseline Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return results


def evaluate_transformer(
    model_dir: str,
    test_df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    max_length: int = 128,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Evaluate transformer model

    Args:
        model_dir: Directory containing saved transformer model
        test_df: Test dataframe
        text_column: Name of text column
        label_column: Name of label column
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating transformer model from {model_dir}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Prepare predictions
    predictions = []
    texts = test_df[text_column].tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)

    y_test = test_df[label_column].values
    y_pred = np.array(predictions)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(f"Transformer Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return results


def evaluate_all_models(config: Dict[str, Any]):
    """
    Evaluate all trained models

    Args:
        config: Configuration dictionary
    """
    # Load test data
    test_df = pd.read_csv(f"{config['data']['processed_path']}/test.csv")

    all_results = {}

    # Evaluate baseline
    logger.info("Evaluating baseline model")
    baseline_results = evaluate_baseline(
        "experiments/outputs/baseline/",
        test_df
    )
    all_results['baseline'] = baseline_results

    # Evaluate transformer models
    transformer_experiments = [
        'distil_roberta_orig',
        'distil_roberta_aug_x2',
        'distil_roberta_aug_x5'
    ]

    for exp_name in transformer_experiments:
        logger.info(f"Evaluating {exp_name}")
        model_dir = f"experiments/outputs/{exp_name}/"

        try:
            results = evaluate_transformer(
                model_dir,
                test_df,
                max_length=config['models']['transformer']['max_length'],
                batch_size=config['models']['transformer']['batch_size']
            )
            all_results[exp_name] = results
        except Exception as e:
            logger.error(f"Error evaluating {exp_name}: {e}")

    # Save all results
    output_path = "experiments/outputs/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"All evaluation results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    results = evaluate_all_models(config)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")

