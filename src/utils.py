"""
Utility functions for event detection project
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "experiments/logs/", log_name: str = "experiment.log"):
    """
    Setup logging configuration

    Args:
        log_dir: Directory to save logs
        log_name: Name of the log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save experiment results to JSON file

    Args:
        results: Dictionary containing results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file

    Args:
        results_path: Path to results file

    Returns:
        Dictionary containing results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

