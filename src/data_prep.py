import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import save_df, load_config, set_seed

cfg = load_config()


def prepare_splits(input_csv, out_dir, test_ratio=0.2, val_ratio=0.1, random_state=42):
    df = pd.read_csv(input_csv)
    # REQUIRED: columns message,lable
    assert "message" in df.columns and "lable" in df.columns
    df = df.dropna(subset=["message", "lable"]).reset_index(drop=True)
    # simple stratified split
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df["lable"], random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_ratio / (1 - test_ratio), stratify=train_val["lable"],
                                  random_state=random_state)
    os.makedirs(out_dir, exist_ok=True)
    save_df(train, os.path.join(out_dir, "train.csv"))
    save_df(val, os.path.join(out_dir, "val.csv"))
    save_df(test, os.path.join(out_dir, "test.csv"))
    print("Saved splits to", out_dir)


if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["random_state"])
    prepare_splits(cfg["dataset_path"], cfg["processed_dir"], cfg["test_ratio"], cfg["val_ratio"], cfg["random_state"])
