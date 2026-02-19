

import pandas as pd
import json
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine
import uuid

# Configuration
DATA_FILE = "data_with_events_roberta.csv"
TAXONOMY_FILE = "icao_taxonomy.json"
OUTPUT_FILE = "classification_results_refined.csv"
EVALUATION_REPORT = "evaluation_report.md"
MODEL_NAME = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.40

# Database Config
DB_USER = "postgres"
DB_PASS = "toormaster"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "aviation"
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Acronym Expansion
ACRONYMS = {
    "NMAC": "Near Mid Air Collision",
    "CFTT": "Controlled Flight Into Terrain",
    "UAS": "Unmanned Aircraft System",
    "RPIC": "Remote Pilot in Command",
    "VMC": "Visual Meteorological Conditions",
    "IMC": "Instrument Meteorological Conditions",
    "TCAS": "Traffic Collision Avoidance System",
    "RA": "Resolution Advisory",
    "TA": "Traffic Advisory"
}

# Hybrid Rules (Outcome keyphrase -> Forced Category Code)
HYBRID_RULES = {
    "birdstrike": "BIRD",
    "bird strike": "BIRD",
    "runway excursion": "RE",
    "ground collision": "GCOL",
    "wildlife": "WILD"
}

def load_data():
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Generate Unique ID if not present
    if "event_id" not in df.columns:
        print("Generating unique 'event_id' for each row...")
        df["event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        
    return df

def load_taxonomy():
    print(f"Loading taxonomy from {TAXONOMY_FILE}...")
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    categories = {}
    for code, details in taxonomy.get("occurrence_categories", {}).items():
        name = details.get("name", "").replace(")", "").strip()
        desc = details.get("description", "").strip()
        
        # Enriched embedding text
        extra_keywords = ""
        if code == "CFIT":
            extra_keywords = " CFTT, Controlled Flight Into Terrain, ground proximity warning"
        elif code == "MAC":
            extra_keywords = " NMAC, Near Mid Air Collision, traffic conflict, TCAS RA, Resolution Advisory, Traffic Advisory"
            
        text_rep = f"Category: {name}. Description: {desc}.{extra_keywords}"
        if "examples" in details and details["examples"]:
            examples = ", ".join(details["examples"])
            text_rep += f" Examples: {examples}."
            
        categories[code] = {
            "name": name,
            "description": desc,
            "text": text_rep
        }
    return categories

def construct_event_string(extracted_events_json):
    try:
        data = json.loads(extracted_events_json)
        qa = data.get("qa_extraction", {}) or {}
        kw = data.get("keyword_extraction", {}) or {}
        
        def get_val(source, key):
            val = source.get(key)
            if isinstance(val, list):
                return ", ".join([str(v) for v in val if v])
            return str(val) if val else ""

        phase = get_val(qa, "PHASE") or get_val(kw, "PHASE")
        actor = get_val(qa, "ACTOR") or get_val(kw, "ACTOR")
        trigger = get_val(qa, "TRIGGER") or get_val(kw, "TRIGGER")
        system = get_val(qa, "SYSTEM") or get_val(kw, "SYSTEM")
        outcome = get_val(qa, "OUTCOME") or get_val(kw, "OUTCOME")
        
        event_str = f"Phase: {phase}. Actor: {actor}. Trigger: {trigger}. System: {system}. Outcome: {outcome}."
        
        # Normalize Acronyms
        for acronym, expansion in ACRONYMS.items():
            # Use regex to replace whole words only
            event_str = re.sub(r'\b' + re.escape(acronym) + r'\b', expansion, event_str, flags=re.IGNORECASE)

        event_str = re.sub(r"\w+: \.", "", event_str)
        event_str = re.sub(r"\s+", " ", event_str).strip()
        
        return event_str, outcome.lower()
    except Exception as e:
        return "", ""

def save_to_db(df, table_name):
    print(f"Saving {table_name} to database...")
    try:
        engine = create_engine(DB_URL)
        # using 'replace' ensures we overwrite the table, preventing duplicates if we run multiple times
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Successfully saved {len(df)} rows to table '{table_name}'.")
    except Exception as e:
        print(f"Database Error: {e}")

def main():
    # 1. Load Resources
    df = load_data()
    categories = load_taxonomy()
    
    # Save original data to DB (with event_id)
    # We save this first to ensure the IDs are established
    save_to_db(df, "original_data")
    
    # 2. Initialize Model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Embed Taxonomy
    print("Embedding taxonomy categories...")
    cat_codes = list(categories.keys())
    cat_texts = [categories[code]["text"] for code in cat_codes]
    cat_embeddings = model.encode(cat_texts, convert_to_tensor=True)
    
    # 4. Process and Classify
    results = []
    
    print("Classifying events...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        event_id = row.get("event_id") # Get the unique ID
        original_cat = row.get("final_category", "OTHER")
        narrative = str(row.get("narrative_1", ""))
        extracted_json = row.get("extracted_events", "{}")
        
        event_str, outcome_text = construct_event_string(extracted_json)
        
        if len(event_str) < 10:
             event_str = str(row.get("synopsis", "")) or narrative[:200]

        # Hybrid Rule Check
        forced_code = None
        for key, code in HYBRID_RULES.items():
            if key in outcome_text or key in event_str.lower():
                forced_code = code
                break
        
        if forced_code:
            predicted_code = forced_code
            predicted_name = categories.get(forced_code, {}).get("name", "Rule Based")
            confidence = 1.0
            method = "Rule"
        else:
            # Embedding Matching
            event_embedding = model.encode(event_str, convert_to_tensor=True)
            cosine_scores = util.cos_sim(event_embedding, cat_embeddings)[0]
            
            top_result = float(np.max(cosine_scores.cpu().numpy()))
            top_idx = int(np.argmax(cosine_scores.cpu().numpy()))
            
            # Threshold Check
            if top_result >= CONFIDENCE_THRESHOLD:
                predicted_code = cat_codes[top_idx]
                predicted_name = categories[predicted_code]["name"]
                confidence = top_result
                method = "Embedding"
            else:
                predicted_code = "UNK" # Unknown / Low Confidence
                predicted_name = "Unknown / Low Confidence"
                confidence = top_result
                method = "Low Confidence"

        results.append({
            "event_id": event_id, # Track back to original record
            "row_id": idx,
            "original_category": original_cat,
            "event_string": event_str,
            "predicted_code": predicted_code,
            "predicted_name": predicted_name,
            "confidence": confidence,
            "method": method
        })

    # 5. Save Results to CSV and DB
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    save_to_db(results_df, "classification_results")
    
    print(f"Results saved to {OUTPUT_FILE}")
    generate_report(results_df)

def generate_report(df):
    print("Generating evaluation report...")
    
    total = len(df)
    other_original = len(df[df["original_category"] == "OTHER"])
    reclassified_mask = (df["original_category"] == "OTHER") & (df["predicted_code"] != "OTHR") & (df["predicted_code"] != "UNK")
    reclassified_count = len(df[reclassified_mask])
    
    unk_count = len(df[df["predicted_code"] == "UNK"])
    mac_count = len(df[df["predicted_code"] == "MAC"])
    
    report = f"""# Evaluation Report: Final Refinement (Thresh={CONFIDENCE_THRESHOLD})

## Overview
- **Total Reports**: {total}
- **Strategy**: Acronyms (incl. TCAS/RA) + UUIDs + Thresh {CONFIDENCE_THRESHOLD}
- **Reclassified "OTHER"**: {reclassified_count} ({reclassified_count/other_original*100:.1f}%)
- **Unclassified (UNK)**: {unk_count} (Reduced from ~112)
- **MAC Detected**: {mac_count}

## Comparison by Method
| Method | Count | Avg Confidence |
|---|---|---|
| Rule (Hybrid) | {len(df[df['method']=='Rule'])} | 1.00 |
| Embedding | {len(df[df['method']=='Embedding'])} | {df[df['method']=='Embedding']['confidence'].mean():.4f} |
| Low Confidence | {len(df[df['method']=='Low Confidence'])} | {df[df['method']=='Low Confidence']['confidence'].mean():.4f} |

## Top High-Confidence Reclassifications
"""
    
    top_reclassified = df[reclassified_mask].sort_values(by="confidence", ascending=False).head(10)
    
    for _, row in top_reclassified.iterrows():
        report += f"""
### Row {row['row_id']} ({row['method']})
- **Event**: `{row['event_string']}`
- **Predicted**: **{row['predicted_code']}** ({row['predicted_name']})
- **Confidence**: {row['confidence']:.4f}
"""

    with open(EVALUATION_REPORT, "w") as f:
        f.write(report)
    print(f"Report saved to {EVALUATION_REPORT}")

if __name__ == "__main__":
    main()

