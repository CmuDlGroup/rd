# Deep Learning Improvement Plan
## Aviation Incident Classification — CMU 18-786

**Dataset**: 19,675 rows (`asn_scraped_accidents`) | **Classes**: 37 ECCAIRS occurrence categories  
**Deadline**: End of April 2026 (~8 weeks)

---

## Current System Diagnosis

| Component | What it actually does | Problem |
|---|---|---|
| Stage 1 — Event Extraction | RoBERTa QA (`deepset/roberta-base-squad2`) extracts 5 spans | Lossy — discards most narrative context |
| Stage 2 — Classification | `all-MiniLM-L6-v2` cosine similarity vs taxonomy embeddings | No task-specific training — pure retrieval |
| Stage 3 — Fine-tuning | `roberta-base` trained on Stage 2 pseudo-labels | Circular: learns to replicate Stage 2 errors |
| Input to Stage 3 | `event_string` (reconstructed from QA spans) | ~90% of narrative text discarded before training |

**Measured outcome**: 32% UNK rate, mean confidence 0.465 (barely above the 0.40 threshold).

---

## Evaluation Protocol (Fixed Across All Experiments)

All experiments below are evaluated against the **same held-out test set** of 2,000 randomly selected rows from `asn_scraped_accidents` (10% of data, stratified by original `category` field). This set is fixed before any experiment begins and never used for training.

### Metrics (report all 6 for every experiment)

| # | Metric | Formula / Tool | Why |
|---|---|---|---|
| M1 | **Macro F1** | `sklearn.metrics.f1_score(average='macro')` | Primary metric — penalises ignoring rare classes |
| M2 | **Weighted F1** | `sklearn.metrics.f1_score(average='weighted')` | Overall accuracy adjusted for class frequency |
| M3 | **UNK Rate** | `len(predicted == 'UNK') / total` | Measures coverage — lower is better |
| M4 | **Mean Confidence** | `mean(confidence)` on non-UNK predictions | Calibration signal |
| M5 | **Top-5 Accuracy** | Predicted category in top-5 softmax outputs | Graceful fallback quality |
| M6 | **Per-class F1** | Full classification_report | Identifies which categories are still weak |

### Baseline Comparisons

Every experiment must report a delta (Δ) against:
- **B0**: Current system (Stage 2 cosine similarity, no training)
- **B_prev**: The immediately preceding experiment

### Experiment Log Format

Each experiment is documented with this template:

```
## EXP-N: [Name]
**Hypothesis**: [what we expect to improve and why]
**Changes**: [what's different from previous experiment]
**Training config**: [epochs, LR, batch size, hardware]
**Results**:
| Metric | B0 (baseline) | B_prev | This exp | Δ vs B0 | Δ vs B_prev |
|---|---|---|---|---|---|
| Macro F1 | | | | | |
| Weighted F1 | | | | | |
| UNK Rate | | | | | |
| Mean Confidence | | | | | |
| Top-5 Acc | | | | | |
**Key observations**: [per-class breakdown, failure cases]
**Decision**: ADVANCE / REVISIT / STOP
```

---

## Experiment Sequence

### EXP-0: Baseline Audit (Week 1)
**Goal**: Establish true baseline numbers on the fixed test set before any changes.

Run the current `classify_events.py` on the 2,000-row test set. This is the reference point for all future deltas. No model changes.

**Deliverable**: Filled EXP-0 results table. If Macro F1 > 0.60, the problem is easier than expected — adjust ambition accordingly.

---

### EXP-1: End-to-End RoBERTa on Raw Narratives (Week 1–2)
**Hypothesis**: Feeding the full narrative text directly to RoBERTa (instead of the lossy `event_string`) will give the model far more signal and substantially reduce the UNK rate.

**Key changes**:
- Input: `narrative_1` (full text, truncated to 512 tokens) → replaces `event_string`
- Model: `roberta-base` fine-tuned with `RobertaForSequenceClassification`
- Training data: high-confidence Stage 2 predictions from `classification_results` table  
  (method = 'Rule' OR (method = 'Embedding' AND confidence ≥ 0.50))
- Loss: `CrossEntropyLoss` with **class weights** (`compute_class_weight('balanced', ...)`)
- Epochs: 5 | LR: 2e-5 | Batch: 16 | Max tokens: 512

**File to modify**: `train_classifier.py`

```python
# Key change — use narrative directly
text = row['narrative_1'][:2000]   # tokeniser will handle truncation
# NOT: row['event_string']
```

**Advance condition**: Macro F1 improves ≥ 5 points over B0 → proceed to EXP-2.

---

### EXP-2: Aviation-Domain Pre-trained Model (Week 2–3)
**Hypothesis**: A model pre-trained on aviation text will have better domain priors, improving especially the rare/confusable categories (CFIT vs LOC-I, ARC vs USOS).

**Key changes**:
- Swap `roberta-base` → `NASA-AIML/MIKA_SafeAeroBERT` (already tested in `compare_models.py`)
- Everything else identical to EXP-1
- Directly compare EXP-1 and EXP-2 on per-class F1 for aviation-specific categories

```python
MODEL_NAME = "NASA-AIML/MIKA_SafeAeroBERT"
```

**Advance condition**: Macro F1 ≥ EXP-1 → proceed. If worse, keep `roberta-base` for EXP-3.

---

### EXP-3: Cross-Encoder Reranking (Week 3–4)
**Hypothesis**: A cross-encoder that jointly encodes `(narrative, category_description)` pairs learns to directly compare the two, which is more powerful than separate bi-encoder embeddings.

**Key changes**:
- New `cross_encoder_classifier.py` script
- Architecture: `RobertaForSequenceClassification` with input:  
  `"[CLS] {narrative[:400]} [SEP] {category_name}: {category_explanation[:200]} [SEP]"`
- Training formulation: **binary** (does this narrative match this category?) with **NT-Xent / contrastive loss**
- Inference: score all 37 categories for each narrative, take argmax
- Training data: same high-confidence pseudo-labels as EXP-1 (positive pairs) + random negatives (negative pairs, 4:1 ratio)

```python
# Positive pair
input_pos = f"{narrative} [SEP] {cat_name}: {cat_explanation[:200]}"
label_pos = 1

# Negative pair (random wrong category)  
input_neg = f"{narrative} [SEP] {wrong_cat_name}: {wrong_cat_explanation[:200]}"
label_neg = 0
```

> [!NOTE]
> Inference is 37× slower (one forward pass per category per sample) — measure and report latency.

**Advance condition**: Macro F1 ≥ best of (EXP-1, EXP-2) → proceed.

---

### EXP-4: Iterative Self-Training (Week 4–6)
**Hypothesis**: Bootstrapping with multiple rounds of pseudo-labeling — where each round uses the previous round's model to generate better labels — will grow and improve the training set iteratively.

**Key changes**:
- New `self_train.py` script implementing the loop:

```
Round 0: high-confidence cosine-similarity labels (current state)
  → train EXP-1/2 model → model_r0.pt

Round 1: run model_r0 on all unlabeled rows
  → keep predictions with softmax confidence ≥ 0.80
  → add to training set → train model_r1.pt

Round 2: run model_r1 on remaining unlabeled rows
  → keep ≥ 0.85 threshold → train model_r2.pt

Repeat until no new high-confidence predictions are added.
```

- Track training set size per round (expect it to grow from ~8K to 15K+)
- Evaluate each round's model on fixed test set — plot learning curve
- **Early stopping**: if Macro F1 does not improve by ≥ 1 point over 2 consecutive rounds, stop

**Deliverable**: Plot of `training_set_size vs Macro_F1` across rounds.

---

### EXP-5: Multi-Label Classification (Week 6–7)
**Hypothesis**: Aviation incidents often involve multiple concurrent event types (e.g. icing + loss of control). A multi-label model produces a richer, more accurate output.

**Key changes**:
- Change output head: `BCEWithLogitsLoss` instead of `CrossEntropyLoss`
- Labels: multi-hot vector of shape `[37]`
- Training data: generate multi-hot labels by taking all ECCAIRS predictions with confidence ≥ 0.45
- Primary metric changes to **mean Average Precision (mAP)** for this experiment only

```python
# Multi-label output
outputs = model(input_ids, attention_mask=attn_mask)
loss = BCEWithLogitsLoss()(outputs.logits, multi_hot_labels.float())

# Inference threshold
predictions = (torch.sigmoid(outputs.logits) > 0.45).int()
```

**Advance condition**: Report results — this is a directional experiment. If mAP improves, recommend multi-label as the final architecture.

---

## Model Selection Decision Tree

```
After EXP-0 (baseline):
  ├─ If UNK Rate < 15% already → re-examine threshold, may not need DL training
  └─ Otherwise → proceed to EXP-1

After EXP-1:
  ├─ Macro F1 improvement ≥ 5pts → ADVANCE to EXP-2
  └─ < 5pts improvement → diagnose: check class weights, try longer training

After EXP-2:
  ├─ AeroBERT ≥ RoBERTa-base → use AeroBERT as backbone for all future exps
  └─ RoBERTa-base better → keep roberta-base

After EXP-3 (cross-encoder):
  ├─ Macro F1 > best bi-encoder → cross-encoder is final Stage 2
  └─ Worse → keep bi-encoder, use EXP-1/2 as final model

After EXP-4 (self-training):
  ├─ Improvement plateaus after N rounds → use round N-1 model
  └─ No improvement → skip; self-training hurts with noisy labels

EXP-5 is additive — run regardless, use if mAP improves.
```

---

## Implementation Files

| File | Owner experiment | Purpose |
|---|---|---|
| `classify_events.py` | EXP-0 baseline | Cosine-similarity classifier (current) |
| `train_classifier.py` | EXP-1, EXP-2 | End-to-end RoBERTa fine-tuner — modify backbone |
| `cross_encoder_classifier.py` | EXP-3 | Cross-encoder training + inference |
| `self_train.py` | EXP-4 | Iterative pseudo-label loop |
| `evaluate.py` | All | Fixed test-set evaluator — always run this after training |
| `experiment_log.md` | All | Running log of all results in the template format above |

---

## `evaluate.py` — Shared Evaluation Script

Must be written **before** any experiment begins. Signature:

```python
def evaluate(model, test_df, label_map, device):
    """
    Returns dict with keys: macro_f1, weighted_f1, unk_rate,
    mean_confidence, top5_acc, per_class_f1_report
    """
```

This script is the single source of truth for all reported numbers. Every experiment calls it on the same 2,000-row test set.

---

## Timeline

| Week | Experiment | Milestone |
|---|---|---|
| 1 | EXP-0 + fixed test set construction | Baseline numbers established |
| 1–2 | EXP-1 | First trained DL model, narrative input |
| 2–3 | EXP-2 | Domain model comparison |
| 3–4 | EXP-3 | Cross-encoder built and evaluated |
| 4–6 | EXP-4 | Self-training loop, learning curve plotted |
| 6–7 | EXP-5 | Multi-label extension |
| 7–8 | Final model selection + report | `experiment_log.md` completed, best model committed |

---

## Key Anti-Patterns to Avoid

> [!WARNING]
> **Do not evaluate on training data.** The test set of 2,000 rows is fixed before EXP-0 and never touched for training.

> [!WARNING]
> **Do not tune hyperparameters on the test set.** Use a separate 10% validation split from the training pool for tuning. Only report test-set numbers after the configuration is locked.

> [!CAUTION]
> **Circular pseudo-labeling.** Each self-training round must use a *higher* confidence threshold than the previous round to prevent the model from reinforcing its own errors.
