import glob
import os
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def summarize_preds(preds_folder):
    files = glob.glob(os.path.join(preds_folder, "preds_seed_*.csv"))
    metrics = []
    for f in files:
        df = pd.read_csv(f)
        y_true = df["label"].values
        y_pred = df["pred"].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
        acc = accuracy_score(y_true, y_pred)
        metrics.append({"file": os.path.basename(f), "accuracy": acc, "precision": p, "recall": r, "f1": f1})
    mdf = pd.DataFrame(metrics)
    summary = mdf[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]).T
    summary.to_csv(os.path.join(preds_folder, "summary_metrics.csv"))
    print("Saved summary to", preds_folder)


if __name__ == "__main__":
    import sys

    summarize_preds(sys.argv[1])
