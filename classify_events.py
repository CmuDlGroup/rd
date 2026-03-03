"""
classify_events.py
Stage 2: ICAO/ECCAIRS taxonomy classification.

Reads incident narratives from the `asn_scraped_accidents` PostgreSQL table,
classifies each using the ECCAIRS occurrence categories from
eccairs_taxonomy_full.json, and writes results to `classification_results`.
"""

import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import create_engine, text
from tqdm import tqdm

from config import DB_URL, SOURCE_TABLE, RESULTS_TABLE, TAXONOMY_FILE, get_raw_conn

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.40
EVALUATION_REPORT = "evaluation_report.md"

# ── Acronym expansion ─────────────────────────────────────────────────────────
ACRONYMS = {
    "NMAC": "Near Mid Air Collision",
    "CFIT": "Controlled Flight Into Terrain",
    "UAS":  "Unmanned Aircraft System",
    "RPIC": "Remote Pilot in Command",
    "VMC":  "Visual Meteorological Conditions",
    "IMC":  "Instrument Meteorological Conditions",
    "TCAS": "Traffic Collision Avoidance System",
    "RA":   "Resolution Advisory",
    "TA":   "Traffic Advisory",
    "LOC":  "Loss of Control",
    "ATC":  "Air Traffic Control",
}

# ── Hybrid keyword rules: phrase → ECCAIRS code ───────────────────────────────
HYBRID_RULES = {
    "birdstrike":       "BIRD",
    "bird strike":      "BIRD",
    "runway excursion":  "RE",
    "ground collision":  "GCOL",
    "wildlife":         "WILD",
    "loss of control":  "LOC-I",
    "fuel exhaustion":  "FUEL",
    "fuel starvation":  "FUEL",
    "icing":            "ICE",
    "wind shear":       "WSTRW",
    "windshear":        "WSTRW",
    "turbulence":       "TURB",
    "evacuation":       "EVAC",
    "runway incursion":  "RI",
    "midair collision":  "MAC",
    "mid-air collision": "MAC",
    "near miss":        "MAC",
}


def load_data():
    """Load narratives from asn_scraped_accidents, drop rows without text."""
    print(f"Loading data from '{SOURCE_TABLE}'...")
    query = f"""
        SELECT
            uid          AS event_id,
            narrative    AS narrative_1,
            category     AS final_category,
            phase
        FROM {SOURCE_TABLE}
        WHERE narrative IS NOT NULL
          AND TRIM(narrative) <> ''
    """
    conn = get_raw_conn()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    print(f"  Loaded {len(df):,} rows with non-empty narratives.")
    return df



def load_taxonomy():
    """
    Parse ECCAIRS occurrence categories from eccairs_taxonomy_full.json.

    Structure: entities[0].attributes[16] is the 'Occurrence category' attribute.
    Each value has:
      - description: "CODE: Human readable name"  (e.g. "CFIT: Controlled flight into or toward terrain")
      - explanation: full ICAO usage notes (rich signal for embedding)
    """
    print(f"Loading taxonomy from '{TAXONOMY_FILE}'...")
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Navigate to the occurrence category attribute (index 16, verified by inspection)
    occ_attr = data["entities"][0]["attributes"][16]
    assert occ_attr["description"] == "Occurrence category", (
        f"Unexpected attribute: {occ_attr['description']!r}. "
        "Check ECCAIRS file structure — the 'Occurrence category' attribute may have moved."
    )

    categories = {}
    for val in occ_attr["values"]:
        raw_desc = val.get("description", "")
        explanation = val.get("explanation", "").strip()

        # description = "CODE: Human name" — split on first colon
        if ":" in raw_desc:
            code, name = raw_desc.split(":", 1)
            code = code.strip()
            name = name.strip()
        else:
            code = raw_desc.strip()
            name = raw_desc.strip()

        # Build rich embedding text: code + name + first 500 chars of usage notes
        text_rep = f"Code: {code}. Category: {name}. {explanation[:500]}"
        categories[code] = {
            "name":        name,
            "description": explanation[:200],
            "text":        text_rep,
        }

    print(f"  Loaded {len(categories)} occurrence categories from ECCAIRS taxonomy.")
    return categories


def construct_event_string(row):
    """
    Build a classification-ready string from a DB row.
    Falls back to first 200 chars of narrative when no text is available.
    """
    narrative = str(row.get("narrative_1", ""))
    phase     = str(row.get("phase", "") or "")

    # Expand known acronyms
    expanded = narrative
    for acronym, expansion in ACRONYMS.items():
        expanded = re.sub(
            r"\b" + re.escape(acronym) + r"\b",
            expansion,
            expanded,
            flags=re.IGNORECASE,
        )

    # Prepend flight phase if known
    if phase and phase.lower() not in ("nan", "none", ""):
        event_str = f"Phase: {phase}. {expanded}"
    else:
        event_str = expanded

    # Collapse whitespace
    event_str = re.sub(r"\s+", " ", event_str).strip()

    # Truncate to 512 tokens worth of chars (~2000 chars)
    if len(event_str) > 2000:
        event_str = event_str[:2000]

    return event_str


def save_to_db(df, table_name):
    print(f"Saving {len(df):,} rows to table '{table_name}'...")
    try:
        engine = create_engine(DB_URL)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"  Done.")
    except Exception as e:
        print(f"  Database error: {e}")


def main():
    engine = create_engine(DB_URL)

    # 1. Load resources
    df         = load_data()
    categories = load_taxonomy()

    # 2. Initialise embedding model
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Embed taxonomy categories once
    print("Encoding taxonomy categories...")
    cat_codes     = list(categories.keys())
    cat_texts     = [categories[code]["text"] for code in cat_codes]
    cat_embeddings = model.encode(cat_texts, convert_to_tensor=True, show_progress_bar=False)

    # 4. Classify each row
    results = []
    print("Classifying events...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        event_id      = row["event_id"]
        original_cat  = str(row.get("final_category", "")) or "OTHER"
        event_str     = construct_event_string(row)
        outcome_lower = event_str.lower()

        # Hybrid keyword rule check (fast path)
        forced_code = None
        for phrase, code in HYBRID_RULES.items():
            if phrase in outcome_lower:
                forced_code = code
                break

        if forced_code:
            predicted_code  = forced_code
            predicted_name  = categories.get(forced_code, {}).get("name", "Rule Based")
            confidence      = 1.0
            method          = "Rule"
        else:
            # Semantic embedding match
            event_emb     = model.encode(event_str, convert_to_tensor=True)
            cosine_scores = util.cos_sim(event_emb, cat_embeddings)[0].cpu().numpy()

            top_idx    = int(np.argmax(cosine_scores))
            top_score  = float(cosine_scores[top_idx])

            if top_score >= CONFIDENCE_THRESHOLD:
                predicted_code = cat_codes[top_idx]
                predicted_name = categories[predicted_code]["name"]
                confidence     = top_score
                method         = "Embedding"
            else:
                predicted_code = "UNK"
                predicted_name = "Unknown / Low Confidence"
                confidence     = top_score
                method         = "Low Confidence"

        results.append({
            "event_id":          event_id,
            "original_category": original_cat,
            "event_string":      event_str[:500],   # cap stored text length
            "predicted_code":    predicted_code,
            "predicted_name":    predicted_name,
            "confidence":        round(confidence, 6),
            "method":            method,
        })

    # 5. Persist
    results_df = pd.DataFrame(results)
    save_to_db(results_df, RESULTS_TABLE)

    # Also save CSV as a local backup
    backup_csv = "classification_results_refined.csv"
    results_df.to_csv(backup_csv, index=False)
    print(f"Backup CSV saved to '{backup_csv}'.")

    generate_report(results_df)


def generate_report(df):
    total       = len(df)
    unk_count   = len(df[df["predicted_code"] == "UNK"])
    rule_count  = len(df[df["method"] == "Rule"])
    emb_df      = df[df["method"] == "Embedding"]
    lc_df       = df[df["method"] == "Low Confidence"]

    emb_conf = emb_df["confidence"].mean() if len(emb_df) else 0
    lc_conf  = lc_df["confidence"].mean()  if len(lc_df)  else 0

    top_codes = df["predicted_code"].value_counts().head(10)
    top_table = "\n".join(
        f"| {code} | {cnt} |" for code, cnt in top_codes.items()
    )

    report = f"""# Classification Evaluation Report

## Overview
- **Total incidents classified**: {total:,}
- **Taxonomy source**: `{TAXONOMY_FILE}` (ECCAIRS 7.1.0.0 — {37} occurrence categories)
- **Data source**: `{SOURCE_TABLE}` (PostgreSQL)

## Method Breakdown
| Method | Count | Avg Confidence |
|---|---|---|
| Rule (keyword) | {rule_count} | 1.0000 |
| Embedding | {len(emb_df)} | {emb_conf:.4f} |
| Low Confidence | {len(lc_df)} | {lc_conf:.4f} |
| **UNK total** | **{unk_count}** | — |

## Top Predicted Categories
| Code | Count |
|---|---|
{top_table}
"""

    with open(EVALUATION_REPORT, "w") as f:
        f.write(report)
    print(f"Evaluation report saved to '{EVALUATION_REPORT}'.")
    print(report)


if __name__ == "__main__":
    main()
