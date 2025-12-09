"""
Data augmentation module for text data
"""

import pandas as pd
import random
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Text data augmentation class"""

    def __init__(self, seed: int = 42):
        """
        Initialize augmenter

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.seed = seed

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words with synonyms

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        # TODO: Implement synonym replacement using WordNet or similar
        return text

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n words

        Args:
            text: Input text
            n: Number of words to insert

        Returns:
            Augmented text
        """
        # TODO: Implement random insertion
        return text

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p

        Args:
            text: Input text
            p: Probability of deleting each word

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text

        new_words = [word for word in words if random.random() > p]

        if len(new_words) == 0:
            return random.choice(words)

        return ' '.join(new_words)

    def back_translation(self, text: str) -> str:
        """
        Augment text using back translation

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        # TODO: Implement back translation using translation models
        # For now, return original text
        return text

    def augment(self, text: str, method: str = 'all') -> str:
        """
        Apply augmentation method

        Args:
            text: Input text
            method: Augmentation method to use

        Returns:
            Augmented text
        """
        if method == 'synonym_replacement':
            return self.synonym_replacement(text)
        elif method == 'random_insertion':
            return self.random_insertion(text)
        elif method == 'random_swap':
            return self.random_swap(text)
        elif method == 'random_deletion':
            return self.random_deletion(text)
        elif method == 'back_translation':
            return self.back_translation(text)
        else:
            # Random choice
            methods = [self.random_swap, self.random_deletion]
            return random.choice(methods)(text)


def augment_dataset(
    df: pd.DataFrame,
    augmentation_factor: int = 2,
    text_column: str = 'text',
    seed: int = 42
) -> pd.DataFrame:
    """
    Augment dataset by specified factor

    Args:
        df: Input dataframe
        augmentation_factor: How many times to augment (e.g., 2 = 2x data, 5 = 5x data)
        text_column: Name of text column
        seed: Random seed

    Returns:
        Augmented dataframe
    """
    logger.info(f"Augmenting dataset by factor of {augmentation_factor}")

    augmenter = TextAugmenter(seed=seed)
    augmented_dfs = [df.copy()]

    for i in range(augmentation_factor - 1):
        logger.info(f"Creating augmentation round {i+1}")
        aug_df = df.copy()
        aug_df[text_column] = aug_df[text_column].apply(augmenter.augment)
        augmented_dfs.append(aug_df)

    result_df = pd.concat(augmented_dfs, ignore_index=True)
    logger.info(f"Augmented dataset size: {len(result_df)}")

    return result_df


if __name__ == "__main__":
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config()

    # Load training data
    train_df = pd.read_csv(f"{config['data']['processed_path']}/train.csv")

    # Create augmented datasets
    for factor in config['augmentation']['augmentation_factors']:
        logger.info(f"Creating {factor}x augmented dataset")
        aug_df = augment_dataset(train_df, augmentation_factor=factor)

        output_path = f"{config['data']['augmented_path']}/train_aug_x{factor}.csv"
        aug_df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

