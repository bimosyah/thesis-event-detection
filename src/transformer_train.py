"""
Transformer model training module
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize dataset

        Args:
            texts: List of text strings
            labels: List of labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_transformer(
    config: Dict[str, Any],
    train_data_path: str,
    output_dir: str,
    experiment_name: str = "transformer"
):
    """
    Train transformer model

    Args:
        config: Configuration dictionary
        train_data_path: Path to training data
        output_dir: Directory to save outputs
        experiment_name: Name of the experiment
    """
    logger.info(f"Starting transformer training: {experiment_name}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(f"{config['data']['processed_path']}/val.csv")

    # Initialize tokenizer and model
    model_name = config['models']['transformer']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine number of labels
    num_labels = train_df['label'].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Create datasets
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=config['models']['transformer']['max_length']
    )

    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=config['models']['transformer']['max_length']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['models']['transformer']['num_epochs'],
        per_device_train_batch_size=config['models']['transformer']['batch_size'],
        per_device_eval_batch_size=config['models']['transformer']['batch_size'],
        learning_rate=config['models']['transformer']['learning_rate'],
        warmup_steps=config['models']['transformer']['warmup_steps'],
        weight_decay=0.01,
        logging_dir=f"{config['training']['log_dir']}/{experiment_name}",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete")


if __name__ == "__main__":
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    # Train on original data
    train_transformer(
        config,
        train_data_path=f"{config['data']['processed_path']}/train.csv",
        output_dir="experiments/outputs/distil_roberta_orig/",
        experiment_name="distil_roberta_orig"
    )

    # Train on augmented data (x2)
    train_transformer(
        config,
        train_data_path=f"{config['data']['augmented_path']}/train_aug_x2.csv",
        output_dir="experiments/outputs/distil_roberta_aug_x2/",
        experiment_name="distil_roberta_aug_x2"
    )

    # Train on augmented data (x5)
    train_transformer(
        config,
        train_data_path=f"{config['data']['augmented_path']}/train_aug_x5.csv",
        output_dir="experiments/outputs/distil_roberta_aug_x5/",
        experiment_name="distil_roberta_aug_x5"
    )

