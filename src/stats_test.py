import glob
import numpy as np
import os
import pandas as pd

from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar


def read_f1s(folder):
    files = glob.glob(os.path.join(folder, "metrics_seed_*.csv"))
    f1s = []
    for f in files:
        df = pd.read_csv(f)
        if "test_f1" in df.columns:
            f1 = df["test_f1"].iloc[0]
        elif "test_f1" in df.columns or "eval_f1" in df.columns:
            # try generic
            f1 = df.filter(like="f1").iloc[0, 0]
        else:
            f1 = df.filter(like="f1").iloc[0]
        f1s.append(f1)
    return np.array(f1s)


def paired_ttest(folder_a, folder_b):
    a = read_f1s(folder_a)
    b = read_f1s(folder_b)
    stat, p = ttest_rel(a, b)
    print("Paired t-test: stat=%.4f p=%.4f" % (stat, p))
    return stat, p


def mcnemar_test(preds_a_folder, preds_b_folder):
    # assumes same test order
    import glob
    fa = sorted(glob.glob(os.path.join(preds_a_folder, "preds_seed_*.csv")))
    fb = sorted(glob.glob(os.path.join(preds_b_folder, "preds_seed_*.csv")))
    # take first seed for McNemar (paired predictions)
    df_a = pd.read_csv(fa[0])
    df_b = pd.read_csv(fb[0])
    assert len(df_a) == len(df_b)
    a_pred = df_a["pred"].values
    b_pred = df_b["pred"].values
    y_true = df_a["lable"].values
    # contingency: [ [both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong] ]
    both_correct = 0;
    a_correct_b_wrong = 0;
    a_wrong_b_correct = 0;
    both_wrong = 0
    for yt, pa, pb in zip(y_true, a_pred, b_pred):
        ca = (pa == yt)
        cb = (pb == yt)
        if ca and cb:
            both_correct += 1
        elif ca and (not cb):
            a_correct_b_wrong += 1
        elif (not ca) and cb:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1
    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    res = mcnemar(table, exact=False)
    print("McNemar stat=%.4f p=%.4f" % (res.statistic, res.pvalue))
    print(table)
    return res


if __name__ == "__main__":
    import sys

    # usage: python stats_test.py folderA folderB
    paired_ttest(sys.argv[1], sys.argv[2])
    mcnemar_test(sys.argv[1], sys.argv[2])
