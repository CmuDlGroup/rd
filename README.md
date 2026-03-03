# Aviation Event Classification Pipeline

A deep learning NLP pipeline for automatically classifying aviation incident reports against the ICAO occurrence taxonomy. The system combines transformer-based information extraction, semantic embedding classification, and fine-tuned RoBERTa to turn raw narrative text into structured, ICAO-coded events.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1 — Extract Events](#step-1--extract-events)
  - [Step 2 — Classify Events](#step-2--classify-events)
  - [Step 3 — Fine-tune RoBERTa](#step-3--fine-tune-roberta)
  - [Utilities](#utilities)
- [Data](#data)
- [Taxonomy](#taxonomy)
- [Model Details](#model-details)
- [Output Files](#output-files)
- [Results](#results)

---

## Overview

Aviation safety reports contain rich narrative descriptions of incidents, but are often tagged with coarse or missing event categories. This project builds an end-to-end pipeline that:

1. **Extracts structured event attributes** (Actor, System, Phase, Trigger, Outcome) from free-text narratives using both keyword matching and transformer-based Question Answering.
2. **Classifies events** against the ICAO occurrence taxonomy using semantic embedding similarity and rule-based overrides.
3. **Fine-tunes a RoBERTa classifier** on pseudo-labeled high-confidence predictions using data augmentation.

---

## Pipeline Architecture

```
Raw Narratives (CSV)
        │
        ▼
┌─────────────────────┐
│   Event Extraction  │  extract_events.py
│  ┌───────────────┐  │
│  │Keyword Match  │  │  Pattern matching over curated keyword lists
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ RoBERTa QA    │  │  deepset/roberta-base-squad2 for span extraction
│  └───────────────┘  │
└─────────────────────┘
        │
        ▼  data_with_events_roberta.csv
┌─────────────────────┐
│  Event Classifier   │  classify_events.py
│  ┌───────────────┐  │
│  │ Hybrid Rules  │  │  Keyword → forced ICAO code (birdstrike, RE, etc.)
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │   Semantic    │  │  all-MiniLM-L6-v2 cosine similarity vs. taxonomy
│  │   Embedding   │  │
│  └───────────────┘  │
└─────────────────────┘
        │
        ▼  classification_results_refined.csv
┌─────────────────────┐
│  RoBERTa Fine-tune  │  train_classifier.py
│  Pseudo-labeling +  │  High-confidence results → training set
│  Data Augmentation  │  Synonym replacement augmenter
└─────────────────────┘
        │
        ▼  best_model_roberta.pt
```

---

## Project Structure

```
project/
├── extract_events.py               # Stage 1: Event attribute extraction
├── classify_events.py              # Stage 2: ICAO taxonomy classification
├── train_classifier.py             # Stage 3: Fine-tune RoBERTa classifier
├── compare_models.py               # Utility: Compare QA models (DistilBERT, RoBERTa, Longformer, AeroBERT)
├── analyze_data.py                 # Utility: Inspect extracted events & taxonomy
├── analyze_unknowns.py             # Utility: Diagnose low-confidence / UNK predictions
├── analyze_coverage.py             # Utility: Coverage analysis of extraction output
│
├── data-1770316648579.csv          # Raw aviation incident reports (input)
├── data_with_events.csv            # Output of early extraction run
├── data_with_events_roberta.csv    # Output of Stage 1 (RoBERTa QA extraction)
├── classification_results.csv      # Output of first classification pass
├── classification_results_refined.csv  # Output of refined Stage 2 classification
├── comparison_results.csv          # Output of multi-model QA comparison
│
├── icao_taxonomy.json              # ICAO occurrence category definitions
├── eccairs_taxonomy_full.json      # ECCAIRS full aviation taxonomy (reference)
│
├── best_model_roberta.pt           # Saved fine-tuned RoBERTa model weights
├── evaluation_report.md            # Auto-generated classification evaluation report
│
└── DeepLearning_Project_Proposal.pdf
```

---

## Requirements

**Python 3.8+** is required. Install all dependencies via pip:

```bash
pip install torch transformers sentence-transformers pandas scikit-learn tqdm sqlalchemy psycopg2-binary numpy
```

Key libraries:
| Library | Purpose |
|---|---|
| `transformers` | RoBERTa QA model (`deepset/roberta-base-squad2`) and fine-tuning |
| `sentence-transformers` | Semantic embedding (`all-MiniLM-L6-v2`) for taxonomy matching |
| `torch` | PyTorch training backend |
| `pandas` | Data loading and CSV I/O |
| `scikit-learn` | Train/val split, classification metrics |
| `sqlalchemy` + `psycopg2` | PostgreSQL persistence |

---

## Setup

### 1. Clone / place the project files
Ensure all `.py` scripts and data files listed above are in the same working directory.

### 2. Database (optional)
`classify_events.py` can persist results to a PostgreSQL database. Edit the config block at the top of the file if needed:

```python
DB_USER = "postgres"
DB_PASS = "yourpassword"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "aviation"
```

If you don't have PostgreSQL, the script will print a database error but will still save results to CSV — no data is lost.

---

## Usage

### Step 1 — Extract Events

Extracts structured event attributes from raw narratives using keyword matching **and** transformer QA.

```bash
python extract_events.py
```

- **Input**: `data-1770316648579.csv` (column: `narrative_1`)
- **Output**: `data_with_events_roberta.csv`

Each row gains an `extracted_events` column containing a JSON object:
```json
{
  "keyword_extraction": { "ACTOR": ["Pilot"], "PHASE": ["Landing"], ...},
  "qa_extraction":      { "ACTOR": "the captain", "PHASE": "on approach", ...}
}
```

The QA model used is `deepset/roberta-base-squad2` — a strong Squad2-trained model that gracefully handles unanswerable questions.

---

### Step 2 — Classify Events

Maps each extracted event to an ICAO occurrence category using a hybrid of rule-based overrides and semantic embedding matching.

```bash
python classify_events.py
```

- **Input**: `data_with_events_roberta.csv`, `icao_taxonomy.json`
- **Output**: `classification_results_refined.csv`, `evaluation_report.md`
- **Database**: Writes `original_data` and `classification_results` tables

**Classification logic:**

1. **Construct event string** — merges QA and keyword results, expands aviation acronyms (NMAC → Near Mid Air Collision, TCAS, RA, etc.)
2. **Hybrid rule check** — hard-coded key phrases force a category (e.g. "birdstrike" → `BIRD`, "runway excursion" → `RE`)
3. **Embedding match** — `all-MiniLM-L6-v2` encodes the event string and compares against all taxonomy category embeddings via cosine similarity
4. **Threshold gate** — predictions with similarity < `0.40` are flagged as `UNK` (low confidence)

---

### Step 3 — Fine-tune RoBERTa

Trains a supervised RoBERTa classifier using **pseudo-labeling**: high-confidence predictions from Stage 2 become training labels.

```bash
python train_classifier.py
```

- **Input**: `classification_results_refined.csv`
- **Output**: `best_model_roberta.pt`

**Training strategy:**
- Filters only Rule-classified rows and Embedding rows with confidence ≥ 0.45
- Applies a `SimpleAugmenter` — synonym replacement to expand training data (×2 augmented copies per sample)
- Fine-tunes `roberta-base` with AdamW optimizer + linear warmup scheduler
- Saves the best checkpoint by validation accuracy
- Default: 5 epochs, batch size 8, LR 2e-5, max sequence length 128

**Hardware**: Automatically uses CUDA if available, otherwise falls back to CPU.

---

### Utilities

#### Compare QA Models
Runs four QA models side-by-side on the raw narratives and measures inter-model agreement (relative to RoBERTa as anchor):

```bash
python compare_models.py
```

Models evaluated: `DistilBERT`, `RoBERTa (SQuAD2)`, `Longformer`, `NASA AeroBERT`
Output: `comparison_results.csv`, `model_metrics.md`

#### Analyze Data
Inspects the distribution of `final_category` and previews extraction output:
```bash
python analyze_data.py
```

#### Analyze Low-Confidence Predictions
Diagnoses why events were classified as `UNK` — distinguishing poor extraction from genuine low semantic similarity:
```bash
python analyze_unknowns.py
```

#### Analyze Extraction Coverage
Reports how many rows have empty keyword/QA extraction results:
```bash
python analyze_coverage.py
```

---

## Data

**Input**: `data-1770316648579.csv`

A CSV file of aviation incident/accident reports. Key columns used:
| Column | Description |
|---|---|
| `narrative_1` | Free-text narrative of the event |
| `synopsis` | Short summary used as fallback when extraction is sparse |
| `final_category` | Original category label (may be `OTHER`) |

The dataset contains over 1,700 reports. Many are labelled `OTHER`, which the pipeline aims to reclassify into specific ICAO codes.

---

## Taxonomy

**`icao_taxonomy.json`** — contains ICAO occurrence categories, each with:
- `name` — human-readable category name
- `description` — detailed definition  
- `examples` — sample event descriptions

Example categories: `CFIT` (Controlled Flight Into Terrain), `MAC` (Mid-Air Collision), `BIRD` (Birdstrike), `RE` (Runway Excursion), `GCOL` (Ground Collision).

The embedding classifier uses an enriched text representation per category: `"Category: {name}. Description: {desc}. Examples: {examples}."` — plus manually injected acronym keywords for commonly confused categories (e.g. CFIT, MAC).

**`eccairs_taxonomy_full.json`** — the full ECCAIRS taxonomy is also included as a reference artifact (~100 MB).

---

## Model Details

| Stage | Model | Purpose |
|---|---|---|
| Event Extraction | `deepset/roberta-base-squad2` | Span extraction QA |
| Taxonomy Matching | `all-MiniLM-L6-v2` | Semantic similarity |
| Supervised Fine-tuning | `roberta-base` | Multi-class classification |
| (Evaluated) | `distilbert-base-cased-distilled-squad` | QA comparison |
| (Evaluated) | `valhalla/longformer-base-4096-finetuned-squad` | QA comparison |
| (Evaluated) | `NASA-AIML/MIKA_SafeAeroBERT` | Domain-specific QA comparison |

---

## Output Files

| File | Description |
|---|---|
| `data_with_events_roberta.csv` | Stage 1 output — narratives with extracted event JSON |
| `classification_results_refined.csv` | Stage 2 output — predicted ICAO codes + confidence |
| `evaluation_report.md` | Auto-generated markdown summary of classification results |
| `comparison_results.csv` | Multi-model QA extraction comparison |
| `model_metrics.md` | Aggregate confidence and inter-model agreement scores |
| `best_model_roberta.pt` | Best fine-tuned RoBERTa checkpoint |

---

## Large Files (Not in Repository)

The following files exceed GitHub's file size limits and are **not tracked by git** (see `.gitignore`):

| File | Size | Reason |
|---|---|---|
| `best_model_roberta.pt` | ~476 MB | Fine-tuned RoBERTa model weights |
| `eccairs_taxonomy_full.json` | ~98 MB | Full ECCAIRS aviation taxonomy reference |

> **To use these files**, either:
> - Re-generate `best_model_roberta.pt` by running `python train_classifier.py` (requires the Stage 2 output CSV), or
> - Obtain them from the project team / shared storage and place them in the project root directory.

---

## Results

The pipeline produces a structured evaluation report (`evaluation_report.md`) automatically after each Stage 2 run. Key metrics tracked:

- **Total reports processed**
- **"OTHER" reclassification rate** — fraction of originally-vague reports successfully assigned a specific ICAO code
- **UNK (low confidence) count** — reduced from ~112 with threshold tuning
- **MAC detections** — mid-air collision events identified
- **Per-method breakdown** — Rule vs. Embedding confidence averages

Classification methods applied per event:
| Method | Trigger | Confidence |
|---|---|---|
| **Rule** | Keyword match in hybrid rules | 1.00 (forced) |
| **Embedding** | Cosine similarity ≥ 0.40 | Model score |
| **Low Confidence** | Cosine similarity < 0.40 | Model score (flagged UNK) |
