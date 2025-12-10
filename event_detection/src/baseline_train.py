import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from utils import load_config

cfg = load_config()


def run_baseline(train_csv, val_csv, test_csv, output_dir):
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(cfg["baseline"]["tfidf"]["ngram_range"]),
                                  max_features=cfg["baseline"]["tfidf"]["max_features"])),
        ("clf", LogisticRegression(C=cfg["baseline"]["lr"]["C"], solver=cfg["baseline"]["lr"]["solver"],
                                   class_weight=cfg["baseline"]["lr"]["class_weight"]))
    ])
    pipe.fit(train["message"].tolist(), train["lable"].tolist())
    y_pred = pipe.predict(test["message"].tolist())
    report = classification_report(test["lable"], y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(test["lable"], y_pred)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(output_dir, "baseline_pipeline.joblib"))
    pd.DataFrame(report).T.to_csv(os.path.join(output_dir, "baseline_classification_report.csv"))
    print("Confusion matrix:\n", cm)
    print("Saved baseline model and report to", output_dir)


if __name__ == "__main__":
    p = cfg["processed_dir"]
    run_baseline(os.path.join(p, "train.csv"), os.path.join(p, "val.csv"), os.path.join(p, "test.csv"),
                 "experiments/outputs/baseline")
