"""
create_test_set.py
Builds and persists the fixed 2,000-row evaluation test set.

Strategy:
  - Source: classification_results table (EXP-0 output after classify_events.py)
  - Filter: keep only high-confidence predictions
      Rule → all kept (confidence = 1.0)
      Embedding → confidence >= 0.50
  - Sample 2,000 rows, stratified by predicted_code
  - Save as `exp_test_set` table (never overwritten after first creation)
  - Save UIDs as exp_test_set_uids.json for reproducibility

Run once after classify_events.py has finished on all rows.
"""

import sys, os, json, random, io
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, RESULTS_TABLE, get_raw_conn

TEST_SET_TABLE  = "exp_test_set"
TEST_SET_FILE   = "exp_test_set_uids.json"
TARGET_SIZE     = 2000
SEED            = 42
MIN_CONF_EMB    = 0.50   # minimum confidence for Embedding rows


def build_test_set():
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. Load full classification_results via psycopg2
    print("Loading classification_results...")
    conn = get_raw_conn()
    try:
        df = pd.read_sql(f"SELECT * FROM {RESULTS_TABLE}", conn)
    finally:
        conn.close()
    print(f"  Total rows: {len(df)}")

    # 2. Filter to high-confidence only
    hc = df[
        (df["method"] == "Rule") |
        ((df["method"] == "Embedding") & (df["confidence"] >= MIN_CONF_EMB))
    ].copy()
    print(f"  High-confidence rows (Rule + Embedding >= {MIN_CONF_EMB}): {len(hc)}")

    # Exclude UNK from test set (no useful label)
    hc = hc[hc["predicted_code"] != "UNK"].copy()
    print(f"  After removing UNK: {len(hc)}")

    # 3. Stratified sample — proportional to class frequency, min 1 per class
    class_counts = hc["predicted_code"].value_counts()
    n_classes    = len(class_counts)
    target       = min(TARGET_SIZE, len(hc))
    print(f"  Sampling {target} rows from {n_classes} categories...")

    sampled_parts = []
    total_hc = len(hc)
    remaining = target

    for code, count in class_counts.items():
        # Proportional share, at least 1
        share = max(1, round(count / total_hc * target))
        share = min(share, count)  # can't take more than available
        part  = hc[hc["predicted_code"] == code].sample(n=share, random_state=SEED)
        sampled_parts.append(part)
        remaining -= share

    test_df = pd.concat(sampled_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Trim or top-up to exactly TARGET_SIZE if off by a few
    if len(test_df) > target:
        test_df = test_df.head(target)
    elif len(test_df) < target:
        # Fill remaining from unused HC rows
        used_ids = set(test_df["event_id"])
        pool = hc[~hc["event_id"].isin(used_ids)]
        extra = pool.sample(n=min(target - len(test_df), len(pool)), random_state=SEED)
        test_df = pd.concat([test_df, extra]).reset_index(drop=True)

    print(f"  Final test set size: {len(test_df)}")
    print(f"  Category distribution:\n{test_df['predicted_code'].value_counts().to_string()}")

    # 4. Save to DB using raw psycopg2 COPY
    conn = get_raw_conn()
    try:
        cur = conn.cursor()
        # Check if table already exists
        cur.execute("SELECT to_regclass(%s)", (TEST_SET_TABLE,))
        exists = cur.fetchone()[0]

        if exists:
            print(f"\nWARNING: '{TEST_SET_TABLE}' already exists — not overwriting.")
            print("Delete the table manually if you need to rebuild it.")
        else:
            type_map = {"object": "TEXT", "float64": "DOUBLE PRECISION", "int64": "BIGINT", "int32": "INTEGER"}
            col_defs = ", ".join(f'"{c}" {type_map.get(str(test_df[c].dtype), "TEXT")}' for c in test_df.columns)
            cur.execute(f'CREATE TABLE "{TEST_SET_TABLE}" ({col_defs})')

            buf = io.StringIO()
            test_df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cols = ", ".join(f'"{c}"' for c in test_df.columns)
            cur.copy_expert(f'COPY "{TEST_SET_TABLE}" ({cols}) FROM STDIN WITH CSV NULL AS \'\\N\'', buf)
            conn.commit()
            print(f"\nSaved {len(test_df)} rows to '{TEST_SET_TABLE}' table.")
    except Exception as e:
        conn.rollback()
        print(f"DB error: {e}")
    finally:
        conn.close()

    # 5. Save UIDs to JSON for reproducibility (always safe to overwrite)
    uids = test_df["event_id"].tolist()
    with open(TEST_SET_FILE, "w") as f:
        json.dump({"seed": SEED, "min_conf_emb": MIN_CONF_EMB,
                   "size": len(uids), "event_ids": uids}, f, indent=2)
    print(f"UIDs saved to '{TEST_SET_FILE}'.")

    return test_df


if __name__ == "__main__":
    build_test_set()
