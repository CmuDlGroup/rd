# Classification Evaluation Report

## Overview
- **Total incidents classified**: 19,479
- **Taxonomy source**: `eccairs_taxonomy_full.json` (ECCAIRS 7.1.0.0 — 37 occurrence categories)
- **Data source**: `asn_scraped_accidents` (PostgreSQL)

## Method Breakdown
| Method | Count | Avg Confidence |
|---|---|---|
| Rule (keyword) | 4152 | 1.0000 |
| Embedding | 1388 | 0.4221 |
| Low Confidence | 13939 | 0.3332 |
| **UNK total** | **13941** | — |

## Top Predicted Categories
| Code | Count |
|---|---|
| UNK | 13941 |
| TURB | 1412 |
| RE | 944 |
| LOC-I | 500 |
| ICE | 432 |
| ARC | 334 |
| EVAC | 328 |
| AMAN | 240 |
| CTOL | 218 |
| WSTRW | 212 |
