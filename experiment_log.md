# Experiment Log — Aviation Incident Classification

**Dataset**: `asn_scraped_accidents` | **Test set**: `exp_test_set` (2,000 rows, fixed, seed=42)  
**Taxonomy**: ECCAIRS 7.1.0.0 — 37 occurrence categories  

> **Label note**: No expert-annotated ground truth exists. EXP-0 Rule/high-confidence predictions serve as silver labels. F1 in EXP-1+ measures agreement with these silver labels AND improvement in UNK coverage.

---

## EXP-0: Baseline Audit (Cosine Similarity Classifier)
**Date**: 2026-03-05  
**Script**: `classify_events.py`  
**Hypothesis**: Establish reference numbers before any DL training.

### Full-corpus results (19,479 rows)
| Method | Count | % | Avg Confidence |
|---|---|---|---|
| Rule (keyword) | 4,152 | 21.3% | 1.0000 |
| Embedding (≥0.40) | 1,388 | 7.1% | 0.4221 |
| **Low Confidence / UNK** | **13,939** | **71.6%** | 0.3332 |

**Global UNK Rate: 71.6%** — the dominant problem to solve.

### Test set (2,000 rows — all high-confidence silver labels)
| Metric | Value |
|---|---|
| Test set size | 2,000 |
| Categories present | 13 of 37 |
| UNK Rate on test set | 0.00% (by construction — test set is high-confidence) |
| Mean Confidence | 0.9993 |
| Min Confidence | 0.5135 |
| Method breakdown | Rule: 1,997 / Embedding: 3 |

### Test set category distribution
| Code | Count | % |
|---|---|---|
| TURB | 679 | 34.0% |
| RE | 434 | 21.7% |
| ICE | 207 | 10.4% |
| LOC-I | 204 | 10.2% |
| EVAC | 147 | 7.4% |
| WSTRW | 102 | 5.1% |
| GCOL | 77 | 3.9% |
| BIRD | 76 | 3.8% |
| FUEL | 44 | 2.2% |
| MAC | 23 | 1.2% |
| WILD | 3 | 0.2% |
| RI | 3 | 0.2% |
| ARC | 1 | 0.1% |

### Key observations
- Test set is **Rule-dominated** (99.9%) — keyword matches (turbulence, runway excursion, icing, etc.) are near-certain, which is why confidence is so high.
- Only **13 of 37** ECCAIRS categories appear in high-confidence results. Categories requiring nuanced reasoning (CFIT, LOC-G, SCF-NP, MAC low-confidence, etc.) are almost entirely in the UNK pool.
- The **71.6% UNK rate on the full corpus** is the primary metric to improve. Any DL model that correctly classifies those 13,939 rows adds direct value.

### EXP-0 Baselines (reference for future Δ)
| Metric | B0 Value |
|---|---|
| Full corpus UNK Rate | **71.6%** |
| Full corpus Macro F1 | N/A (no ground truth on UNK rows) |
| Test set Mean Confidence | 0.9993 |
| Active categories | 13 / 37 |

**Decision**: ADVANCE → EXP-1

---

## EXP-1: End-to-End RoBERTa on Raw Narratives
**Status**: ⏳ Not started  
**Hypothesis**: Training RoBERTa directly on full narrative text (instead of lossy event_string) with class-weighted loss will dramatically reduce UNK rate and improve per-class recall.

| Metric | B0 (baseline) | EXP-1 | Δ vs B0 |
|---|---|---|---|
| Full corpus UNK Rate | 71.6% | — | — |
| Macro F1 (vs silver labels) | 1.000* | — | — |
| Weighted F1 (vs silver labels) | 1.000* | — | — |
| Mean Confidence | 0.9993 | — | — |
| Top-5 Accuracy | — | — | — |

*trivially 1.0 since test set IS the silver labels

---

## EXP-2: Aviation-Domain Pre-trained Model (AeroBERT)
**Status**: ⏳ Not started

| Metric | B0 | EXP-1 | EXP-2 | Δ vs B0 | Δ vs EXP-1 |
|---|---|---|---|---|---|
| Full corpus UNK Rate | 71.6% | — | — | — | — |
| Macro F1 | — | — | — | — | — |
| Weighted F1 | — | — | — | — | — |
| Mean Confidence | 0.9993 | — | — | — | — |

---

## EXP-3: Cross-Encoder Reranking
**Status**: ⏳ Not started

---

## EXP-4: Iterative Self-Training
**Status**: ⏳ Not started

---

## EXP-5: Multi-Label Classification
**Status**: ⏳ Not started
