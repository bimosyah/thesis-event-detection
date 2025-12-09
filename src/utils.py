import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def set_seed(seed):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_df(df, path):
    df.to_csv(path, index=False)
