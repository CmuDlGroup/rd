"""
evaluate.py
Shared evaluation script for all experiments.

Usage:
    Directly:  python evaluate.py
    Imported:  from evaluate import evaluate, load_test_set

Loads the fixed test set from the `exp_test_set` DB table, runs model
predictions, and reports all 6 standard metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sqlalchemy import create_engine

from config import DB_URL, RESULTS_TABLE, get_raw_conn

TEST_SET_TABLE = "exp_test_set"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_set():
    """Load the fixed 2,000-row test set with silver labels."""
    conn = get_raw_conn()
    try:
        df = pd.read_sql(f"SELECT * FROM {TEST_SET_TABLE}", conn)
    finally:
        conn.close()
    print(f"Loaded {len(df)} test-set rows.")
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, confidences=None, top5_preds=None, label_list=None):
    """
    Compute all 6 standard EXP metrics.

    Parameters
    ----------
    y_true : list[str]       silver-label predicted_code from EXP-0
    y_pred : list[str]       this experiment's predicted_code
    confidences : list[float] optional — confidence scores for this experiment
    top5_preds : list[list[str]] optional — top-5 predicted codes per row
    label_list : list[str]   optional — full label set (for zero-count classes)

    Returns
    -------
    dict with keys: macro_f1, weighted_f1, unk_rate, mean_confidence,
                    top5_acc, report (str)
    """
    # Filter UNK out of F1 calc (UNK is not a real category)
    mask = np.array(y_true) != "UNK"
    y_true_f = np.array(y_true)[mask]
    y_pred_f = np.array(y_pred)[mask]

    macro_f1    = f1_score(y_true_f, y_pred_f, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)
    unk_rate    = (np.array(y_pred) == "UNK").mean()

    mean_conf = float(np.mean(confidences)) if confidences is not None else None

    if top5_preds is not None:
        top5_hits = sum(t in p5 for t, p5 in zip(y_true, top5_preds))
        top5_acc  = top5_hits / len(y_true)
    else:
        top5_acc = None

    labels = sorted(set(y_true_f) | set(y_pred_f))
    report = classification_report(y_true_f, y_pred_f, labels=labels,
                                   zero_division=0, digits=3)

    return {
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "unk_rate":    round(float(unk_rate), 4),
        "mean_confidence": round(mean_conf, 4) if mean_conf is not None else "N/A",
        "top5_acc":    round(top5_acc, 4) if top5_acc is not None else "N/A",
        "report":      report,
    }


def print_metrics(metrics, exp_name=""):
    header = f"=== {exp_name} Evaluation Results ===" if exp_name else "=== Evaluation Results ==="
    print(f"\n{header}")
    print(f"  Macro F1       : {metrics['macro_f1']}")
    print(f"  Weighted F1    : {metrics['weighted_f1']}")
    print(f"  UNK Rate       : {metrics['unk_rate']:.2%}")
    print(f"  Mean Confidence: {metrics['mean_confidence']}")
    print(f"  Top-5 Accuracy : {metrics['top5_acc']}")
    print(f"\n  Per-class report:\n{metrics['report']}")


# ── EXP-0 baseline: evaluate current classifier output vs itself ──────────────

def evaluate_exp0():
    """
    For EXP-0 the 'predictions' ARE the silver labels (since the test set is
    built from the EXP-0 output). We report coverage/distribution metrics.
    """
    test_df = load_test_set()

    y_silver = test_df["predicted_code"].tolist()
    confs    = test_df["confidence"].tolist()

    # UNK rate (silver labels already filtered, but count for reference)
    unk_rate = (test_df["predicted_code"] == "UNK").mean()

    print("\n=== EXP-0: Baseline Audit (Silver Label Distribution) ===")
    print(f"  Test set size  : {len(test_df)}")
    print(f"  UNK Rate       : {unk_rate:.2%}")
    print(f"  Mean Confidence: {np.mean(confs):.4f}")
    print(f"  Min  Confidence: {np.min(confs):.4f}")
    print(f"  Max  Confidence: {np.max(confs):.4f}")

    print(f"\n  Method breakdown:")
    print(test_df["method"].value_counts().to_string())

    print(f"\n  Category distribution (top 20):")
    print(test_df["predicted_code"].value_counts().head(20).to_string())

    # Self-agreement: trivially 1.0 for EXP-0 — but compute for future reference
    metrics = compute_metrics(
        y_true=y_silver,
        y_pred=y_silver,
        confidences=confs,
    )
    print(f"\n  [Self-agreement on EXP-0 labels: Macro F1 = {metrics['macro_f1']} (trivially 1.0 for non-UNK rows)]")
    return metrics


if __name__ == "__main__":
    evaluate_exp0()
