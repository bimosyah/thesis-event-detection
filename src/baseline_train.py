"""
Baseline model training module
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline text classification model"""

    def __init__(
        self,
        model_type: str = 'naive_bayes',
        max_features: int = 5000
    ):
        """
        Initialize baseline model

        Args:
            model_type: Type of model ('naive_bayes', 'logistic_regression', 'svm')
            max_features: Maximum number of features for TF-IDF
        """
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train: pd.Series, y_train: pd.Series):
        """
        Train the baseline model

        Args:
            X_train: Training texts
            y_train: Training labels
        """
        logger.info(f"Training {self.model_type} model")

        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_vec, y_train)

        logger.info("Training complete")

    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input texts

        Returns:
            Predicted labels
        """
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def save(self, output_dir: str):
        """
        Save model and vectorizer

        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving model to {output_dir}")
        joblib.dump(self.model, f"{output_dir}/model.pkl")
        joblib.dump(self.vectorizer, f"{output_dir}/vectorizer.pkl")

    def load(self, output_dir: str):
        """
        Load model and vectorizer

        Args:
            output_dir: Directory containing saved model
        """
        logger.info(f"Loading model from {output_dir}")
        self.model = joblib.load(f"{output_dir}/model.pkl")
        self.vectorizer = joblib.load(f"{output_dir}/vectorizer.pkl")


def train_baseline(config: Dict[str, Any], output_dir: str = "experiments/outputs/baseline/"):
    """
    Train baseline model

    Args:
        config: Configuration dictionary
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(f"{config['data']['processed_path']}/train.csv")
    val_df = pd.read_csv(f"{config['data']['processed_path']}/val.csv")

    # Initialize and train model
    baseline = BaselineModel(
        model_type=config['models']['baseline']['type'],
        max_features=config['models']['baseline']['max_features']
    )

    baseline.train(train_df['text'], train_df['label'])

    # Save model
    baseline.save(output_dir)

    logger.info("Baseline model training complete")


if __name__ == "__main__":
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    train_baseline(config)

