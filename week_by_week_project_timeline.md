# Week-by-Week Project Timeline (with Deliverables)

This timeline assumes a **6-week execution window**, suitable for a course project or short research sprint. The structure emphasizes early validation, parallel progress, and continuous documentation.

---

## Week 1 – Scope Lock & Foundations

**Objectives**
- Eliminate scope ambiguity
- Establish shared technical and conceptual ground

**Key Activities**
- Finalize one-sentence project scope and non-goals
- Freeze event schema (ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)
- Draft 1-page annotation and schema guideline
- Select datasets and extract initial OTHER subset (50–100 reports)

**Deliverables**
- Locked scope statement
- Event schema + annotation guideline (v1)
- Curated OTHER report subset

**Owner Focus**
- All team members (alignment-critical)

---

## Week 2 – Event Extraction Baseline

**Objectives**
- Prove that meaningful events can be extracted from OTHER reports

**Key Activities**
- Design BIO tagging scheme for aviation events
- Implement baseline event extractor (BERT / RoBERTa + CRF)
- Run extraction on initial OTHER subset
- Perform qualitative error analysis

**Deliverables**
- Event extraction model (baseline)
- Sample extracted event JSONs
- Precision-focused extraction metrics
- Error analysis notes

**Owner Focus**
- Event Extraction Lead (primary)
- Evaluation Lead (support)

---

## Week 3 – Causal Graph Construction

**Objectives**
- Translate extracted events into interpretable causal structure

**Key Activities**
- Define graph schema (node types, edge types, constraints)
- Implement rule-based DAG construction
- Validate graphs with domain sanity checks
- Generate graph visualizations for sample reports

**Deliverables**
- Graph schema specification
- Causal graph constructor
- Example graphs with explanations

**Owner Focus**
- Graph & Reasoning Lead (primary)
- Event Extraction Lead (support)

---

## Week 4 – Graph-Based Reasoning & Classification

**Objectives**
- Learn incident representations from causal graphs

**Key Activities**
- Implement GNN (GAT or HGT)
- Train graph-level embedding model
- Map embeddings to ADREP categories
- Establish confidence thresholds for reclassification

**Deliverables**
- Graph reasoning model
- Initial reclassification results on OTHER subset
- Confidence threshold policy

**Owner Focus**
- Graph & Reasoning Lead (primary)
- Evaluation Lead (support)

---

## Week 5 – Evaluation & Robustness Analysis

**Objectives**
- Quantify impact and validate safety alignment

**Key Activities**
- Measure OTHER reduction rate
- Compare text-only vs graph-only vs hybrid performance
- Run robustness tests (paraphrased or incomplete narratives)
- Measure latency overhead

**Deliverables**
- Evaluation tables and plots
- Ablation study results
- Robustness and latency analysis

**Owner Focus**
- Evaluation & Integration Lead (primary)
- All team members (interpretation)

---

## Week 6 – Documentation & Finalization

**Objectives**
- Turn results into a defensible research artifact

**Key Activities**
- Finalize Method and Evaluation sections
- Prepare system architecture diagram
- Write limitations and failure mode analysis
- Align narrative with aviation safety analyst reasoning

**Deliverables**
- Final project report / paper draft
- Architecture and pipeline diagrams
- Reproducibility checklist

**Owner Focus**
- All team members

---

## Continuous Activities (All Weeks)

- Weekly sync meetings (Monday)
- Mid-week demos (Wednesday)
- Documentation and decision locking (Friday)
- Version control discipline and experiment logging

---

## Milestone Summary

- **End of Week 2:** Event extraction feasibility proven
- **End of Week 3:** Interpretable causal graphs available
- **End of Week 4:** Graph-based reclassification operational
- **End of Week 5:** Quantified reduction in OTHER category
- **End of Week 6:** Submission-ready research artifact

