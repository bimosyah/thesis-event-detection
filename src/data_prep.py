"""
Data preparation and preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file

    Args:
        file_path: Path to raw CSV file

    Returns:
        DataFrame containing the raw data
    """
    logger.info(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} samples")
    return df


def clean_text(text: str) -> str:
    """
    Clean and preprocess text

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # TODO: Implement text cleaning
    # - Remove URLs
    # - Remove special characters
    # - Lowercase
    # - Remove extra whitespace
    return text.strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset

    Args:
        df: Raw dataframe

    Returns:
        Preprocessed dataframe
    """
    logger.info("Preprocessing data")
    df = df.copy()

    # Clean text column (assuming column name is 'text')
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    logger.info(f"Preprocessed data: {len(df)} samples")
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets

    Args:
        df: Input dataframe
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data into train/val/test sets")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df['label'] if 'label' in df.columns else None
    )

    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed/"
):
    """
    Save train/val/test splits to CSV files

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        output_dir: Directory to save the splits
    """
    logger.info(f"Saving data splits to {output_dir}")

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    logger.info("Data splits saved successfully")


if __name__ == "__main__":
    # Example usage
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    # Load and preprocess data
    df = load_raw_data(config['data']['raw_path'])
    df = preprocess_data(df)

    # Split and save
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config['data']['split_ratio']['train'],
        val_ratio=config['data']['split_ratio']['val'],
        test_ratio=config['data']['split_ratio']['test'],
        random_seed=config['data']['random_seed']
    )

    save_splits(train_df, val_df, test_df, config['data']['processed_path'])

