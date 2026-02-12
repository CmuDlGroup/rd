# ADREP Causal Graph Reasoning: TA Critique & Feasibility Analysis

**Course:** 18-786 Deep Learning  
**Carnegie Mellon University**  
**Review Date:** February 3, 2026  
**Project Deadline:** End of April 2026 (~12 weeks)

---

## Executive Summary

This is an **ambitious and well-motivated project** that tackles a real-world problem with clear impact potential. However, the **scope is too broad for a semester project**, particularly given the 12-week timeline. The proposal attempts to integrate **5+ complex ML components** (NER, graph construction, GAT/HGT, multi-LLM ensemble, consensus engine) into a production-grade system, which would be challenging even for experienced ML engineers.

**Recommendation:** **MAJOR REVISION REQUIRED** - Significantly narrow scope to focus on core innovation (graph reasoning), simplify evaluation, and establish clearer baselines.

**Feasibility Rating:** 3.5/10 as currently proposed; 7.5/10 with recommended modifications

---

## Part 1: Educational Criticism

### 1.1 Problem Formulation & Motivation ⭐⭐⭐⭐

**Strengths:**

- Excellent real-world motivation with clear stakeholder (civil aviation authorities)
- Well-cited background including the SDCPS baseline system
- Specific quantitative gap identified (26.94% "OTHER" rate)
- Clear connection between technical approach (causal graphs) and problem (handling causal ambiguities)

**Concerns:**

**C1.1 - Lack of Baseline Isolation:**  
You're proposing to add graph reasoning as an "8th consensus rule" to an existing 7-rule system. This makes it **scientifically difficult to isolate the contribution** of your graph-based approach.

> **TA Question:** If your final system achieves 95% accuracy, how will you determine how much came from the graphs vs. the existing LLM ensemble? You need ablation studies, but your proposal doesn't mention them.

**C1.2 - "OTHER" Rate as Primary Metric:**  
Reducing the "OTHER" category is important, but be careful - you could reduce it to 0% by randomly assigning categories! You need to ensure:

- The reduction maintains or improves overall accuracy
- The new classifications are actually correct (but see evaluation concerns below)
- You're not just trading one problem for another

**Recommendation:**

```
Add explicit ablation study plan:
1. Existing 7-rule system alone (baseline)
2. Graph reasoning alone on same data
3. Combined system (proposed)
4. Report delta improvements for each component
```

---

### 1.2 Data & Evaluation Strategy ⭐⭐ (Major Concerns)

**Critical Issue - Circular Evaluation:**

This is the **most serious methodological flaw** in your proposal. You state:

> "3,068 silver-labeled for supervised training via GPT-4o CoT prompting"

And then evaluate using:

> "supervised accuracy (>92%, Longformer's agreement with GPT-4o-generated silver labels)"

**This is circular reasoning!** You're training on GPT-4o labels and evaluating against GPT-4o labels. This doesn't tell you if your classifications are **actually correct**, only if they agree with GPT-4o.

**TA Question:** What happens if GPT-4o is systematically wrong on certain incident types? Your system would learn to replicate those errors and be rewarded for it in evaluation.

**Real-World Concern:**  
If you deploy this to Rwanda Civil Aviation Authority (RCAA), they need ground truth accuracy, not agreement with an LLM that might have its own biases.

**Recommended Fix:**

```
Option A (Gold Standard - Ideal):
- Get 200-500 reports manually labeled by aviation domain experts
- Use these for held-out test set evaluation
- Use silver labels only for training

Option B (Pragmatic - If no experts available):
- Use inter-rater reliability between multiple labelers
  (GPT-4o, Claude, Gemini, your system)
- Report Fleiss' Kappa or Krippendorff's Alpha
- Clearly acknowledge this is not ground truth
- Manually inspect 50-100 disagreement cases

Option C (Academic):
- Treat this as an unsupervised/semi-supervised problem
- Evaluate on downstream tasks (e.g., trend detection)
- Use intrinsic metrics (graph quality, entity extraction F1)
```

**Data Concerns:**

1. **Class Imbalance:** You mention OTHER (26.94%) and CFIT (20.78%) but don't describe your handling strategy. Will you:
   - Use class weights in loss function?
   - Oversample minority classes?
   - Use focal loss?

2. **Silver Label Confidence:** You mention ">85% confidence" - what happens to the 15% below threshold?

3. **Dataset Mixing:** Combining ASN (3K), NASA ASRS (24K), NTSB (44K), FAA - these have **different reporting standards and granularity**. Have you analyzed distribution shift between sources?

---

### 1.3 Technical Approach ⭐⭐⭐

**Strengths:**

- Clear 4-layer architecture description
- Appropriate choice of Aviation-BERT-NER for domain specificity
- Good heterogeneous graph design with typed entities
- GAT/HGT are reasonable choices for graph reasoning

**Concerns:**

**C1.3.1 - Entity Extraction Ambiguity:**

You define entities as `(ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)`, but aviation reports are narratives with:

- Ambiguous temporal ordering
- Implicit causality
- Multiple interleaved event chains

**Example:**

> "During final approach in icing conditions, the pilot noticed decreasing engine power. The aircraft descended below glideslope before recovering at 200 feet."

What's the causal graph here?

- Ice → Power Loss → Altitude Deviation?
- Pilot Action → Recovery?
- Ice → Pilot Distraction → Deviation?

**TA Question:** How will you extract causal edges, not just entities? You mention "causal triples like (Ice, caused, Engine Stall)" but don't describe the extraction methodology. Options:

1. **Dependency Parsing** - Extract subject-verb-object + causality markers
2. **LLM-based Relation Extraction** - Prompt for structured triples
3. **Rule-based Patterns** - Match "due to", "resulted in", "led to"
4. **Learned Relation Classifier** - Train on annotated examples

_You need to specify which approach you'll use._

**C1.3.2 - Graph Construction Challenges:**

- **DAG Assumption:** Real accidents may have cycles (e.g., "spatial disorientation → incorrect inputs → worsening disorientation"). How will you handle this?
- **Missing Edges:** If NER misses an entity, the graph structure changes. What's your error propagation analysis?

- **Graph Size Variability:** Simple reports → 3-5 nodes, complex ones → 20+ nodes. How will GAT handle this variance?

**C1.3.3 - GAT vs HGT Choice:**

You mention both GAT and HGT but don't justify the choice:

| Model   | Pros                                     | Cons                    | Best For              |
| ------- | ---------------------------------------- | ----------------------- | --------------------- |
| **GAT** | Simpler, faster, interpretable attention | Ignores edge types      | Homogeneous graphs    |
| **HGT** | Handles heterogeneous edges/nodes        | More parameters, slower | Multi-typed relations |

Given your heterogeneous graph (different entity types + relation types), **HGT is more appropriate**, but it's also more complex to implement and tune.

**Recommendation:** Start with simpler **Graph Convolutional Network (GCN)** baseline, then try GAT, then HGT if time permits.

**C1.3.4 - Integration with LLM Ensemble:**

You describe adding graph reasoning as an "8th rule" checking structural coherence:

> "rejecting 'Landing' graph with 'Takeoff Incident' prediction"

But you don't specify:

1. **How is the check implemented?** Rule-based? Learned?
2. **What if graph says EQUIP but LLMs say LOC-I?** Who wins?
3. **Confidence weighting?** Should graph output be weighted equally with 7 LLM rules?

This needs a **clear fusion strategy**:

```python
# Option A: Hard Voting with Veto
if graph_check_fails(graph, llm_prediction):
    return "OTHER"
else:
    return llm_consensus

# Option B: Weighted Ensemble
final_logits = 0.7 * llm_ensemble_logits + 0.3 * gat_logits
return argmax(final_logits)

# Option C: Cascaded Refinement
initial_pred = llm_ensemble()
refined_pred = graph_refiner(initial_pred, graph)
return refined_pred
```

**You need to specify which strategy and justify it.**

---

### 1.4 Evaluation Metrics ⭐⭐⭐⭐

**Strengths:**

- Comprehensive metric suite (accuracy, F1, latency, agreement)
- Appropriate use of macro-F1 for imbalanced data
- Latency consideration for production deployment
- Statistical significance testing mentioned

**Concerns:**

**C1.4.1 - Explainability Claims:**

You emphasize "explainability" but don't define how you'll **evaluate** it. Explainability is not binary. You need:

1. **Attention Visualization:** Show which graph nodes GAT attends to (good!)
2. **Human Evaluation:** Can RCAA analysts understand the graphs? (missing!)
3. **Faithfulness:** Do attention weights actually reflect model reasoning or are they post-hoc? (critical for XAI)

**Recommended Addition:**

```
Explainability Evaluation:
- Generate graphs for 50 test cases
- Have 2-3 domain experts rate (1-5 scale):
  - Completeness: Are all relevant entities captured?
  - Coherence: Do causal links make sense?
  - Actionability: Can this guide safety interventions?
- Report inter-rater agreement (Krippendorff's α)
```

**C1.4.2 - "OTHER" Reduction Without Quality Check:**

Your success criterion is:

> "OTHER" rate (<30% threshold)

But you don't check **whether the new classifications are correct**. You could achieve 0% "OTHER" by randomly assigning CFIT/LOC-I/etc.

**Add:**

- Precision@K for "recovered from OTHER" cases
- Manual inspection of 100 random "OTHER → classified" transitions

---

### 1.5 Implementation & Infrastructure ⭐⭐

**Concerns:**

**C1.5.1 - Technology Stack Complexity:**

You list: SpaCy, PyTorch Geometric, NetworkX, Neo4j, LangGraph, FastAPI, Django

That's **8 different frameworks**. For a semester project, this is extremely ambitious.

**Recommendation:** Prioritize core ML components, defer production deployment:

```
Phase 1 (Research Prototype):
- PyTorch + PyTorch Geometric for models
- NetworkX for graph viz (skip Neo4j for now)
- Simple Jupyter notebook pipeline

Phase 2 (If time permits):
- FastAPI endpoint
- Skip Django dashboard unless you have web dev experience
```

**C1.5.2 - Latency Claims:**

You claim to "maintain 2.15s latency" while adding:

- NER model forward pass
- Graph construction
- GAT/HGT forward pass

**TA Question:** Have you profiled where the current 2.15s is spent? What's your latency budget for each added component?

Typical timings:

- BERT-NER (512 tokens): ~50-100ms
- Graph construction: ~10-50ms
- GAT forward pass (10 nodes): ~20-50ms
- **BUT** if you batch inefficiently or use CPU, this balloons

**Add:** Computational complexity analysis & profiling plan

---

## Part 2: Feasibility Assessment

### 2.1 Timeline Analysis (12 weeks until end of April)

Let me break down the **realistic timeline** for each component:

| Component                         | Estimated Time | Confidence | Risks                                 |
| --------------------------------- | -------------- | ---------- | ------------------------------------- |
| **Data preprocessing & EDA**      | 1 week         | High       | Format inconsistencies across sources |
| **Aviation-BERT-NER fine-tuning** | 1.5 weeks      | Medium     | May need custom entity annotation     |
| **Causal relation extraction**    | 2 weeks        | **LOW**    | No annotated training data mentioned  |
| **Graph construction pipeline**   | 1 week         | Medium     | DAG validation logic                  |
| **GAT/HGT implementation**        | 2 weeks        | Medium     | Debugging heterogeneous graphs        |
| **LLM ensemble integration**      | 1 week         | High       | You have existing baseline            |
| **Fusion strategy tuning**        | 1 week         | Medium     | Hyperparameter search                 |
| **Evaluation & experiments**      | 1.5 weeks      | High       | Assuming infra is ready               |
| **Ablation studies**              | 1 week         | Medium     | Need multiple trained models          |
| **Report & visualization**        | 1 week         | High       | -                                     |
| **Buffer for debugging**          | 1 week         | -          | **Always needed!**                    |

**Total: ~13 weeks** - Already over budget!

### 2.2 What Can Be Cut?

To make this feasible, **cut or defer**:

1. ❌ **Neo4j deployment** - Use NetworkX + matplotlib
2. ❌ **FastAPI microservice** - Python scripts are fine for research
3. ❌ **Django dashboard** - Use Jupyter widgets or Streamlit
4. ❌ **HGT** - Start with GCN, maybe try GAT
5. ⚠️ **Custom NER training** - Use off-the-shelf model initially, fine-tune if time permits
6. ⚠️ **Multiple LLM ensemble** - Use 1-2 LLMs, not 3+ with temperature variations

### 2.3 Recommended Scope Reduction

**Option A: Focus on Graph Reasoning Core**

```
Simplified Project:
1. Use existing SDCPS preprocessing & silver labels
2. Implement rule-based causal extraction (dependency parsing)
3. Build GCN-based graph classifier
4. Compare: LLM-only vs Graph-only vs Ensemble
5. Analyze where graphs help/hurt (error analysis)
6. Deliver: Jupyter notebook + 8-page report
```

**Option B: Focus on Explainability**

```
Simplified Project:
1. Take existing SDCPS system as black box
2. Build post-hoc explanations via causal graphs
3. Extract graphs from correctly classified reports
4. Human evaluation of explanation quality
5. Deliver: Explanation interface + user study results
```

**I recommend Option A** - it maintains the core technical innovation while being achievable.

---

## Part 3: Prerequisites & Learning Requirements

### 3.1 What Students MUST Already Know

**Foundational (Non-negotiable):**

- ✅ Deep Learning fundamentals (backprop, optimization, loss functions)
- ✅ PyTorch or TensorFlow (tensor operations, autograd, training loops)
- ✅ Transformers architecture (attention mechanism, BERT/RoBERTa)
- ✅ NLP basics (tokenization, embeddings, sequence labeling)
- ✅ Python programming (OOP, data structures, debugging)
- ✅ Version control (Git/GitHub for collaboration)

**Recommended (Doable without, but harder):**

- ⚠️ Graph theory basics (nodes, edges, DAGs, adjacency matrices)
- ⚠️ Named Entity Recognition (BIO tagging, CRF layers)
- ⚠️ Evaluation metrics (precision/recall/F1, macro vs micro averaging)

**Nice to Have:**

- Cloud computing / GPU management
- REST APIs (if doing deployment)
- Domain knowledge of aviation safety (can be learned)

### 3.2 What Students Need to LEARN for This Project

Here's what you'll need to learn **from scratch** or **deepen significantly**:

#### 3.2.1 Graph Neural Networks (2-3 weeks learning curve)

**Core Concepts:**

- Message passing framework
- Graph convolutions (GCN, GraphSAGE)
- Attention mechanisms on graphs (GAT)
- Heterogeneous graphs (HGT)

**Resources:**

1. **CS224W (Stanford)** - Lecture videos on YouTube (weeks 1-4)
   - Watch: "GNN basics", "GCN", "GAT", "Heterogeneous graphs"
2. **Distill.pub** - "A Gentle Introduction to Graph Neural Networks"
3. **PyTorch Geometric Tutorials** - Official documentation

**Practical:**

```python
# You'll need to master:
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, HGTConv

# Building custom GNN layers
# Handling variable-size graphs in batching
# Debugging attention weights
```

**Estimated Time:** 20-25 hours (including coding exercises)

#### 3.2.2 Named Entity Recognition & Relation Extraction (1-2 weeks)

**Core Concepts:**

- Sequence labeling (BIO/BILUO tagging)
- Conditional Random Fields (CRF)
- SpaCy custom pipelines
- Relation extraction approaches

**Resources:**

1. **Hugging Face NER Tutorial** - Token classification guide
2. **SpaCy Course** - Free online course (chapters 2-3)
3. **Papers:** "A Survey on Deep Learning for Named Entity Recognition" (Li et al., 2020)

**Practical:**

```python
# You'll need to:
- Fine-tune transformers.AutoModelForTokenClassification
- Build custom SpaCy pipelines
- Extract relations from dependency parses
```

**Estimated Time:** 15-20 hours

#### 3.2.3 Causal Reasoning & Knowledge Graphs (1 week)

**Core Concepts:**

- Causal vs correlational relationships
- Knowledge graph construction
- Triple extraction (subject, predicate, object)
- Graph validation (DAG checking, cycle detection)

**Resources:**

1. **Book:** "Causal Inference in Statistics: A Primer" (Pearl) - Ch 1-2
2. **Tutorial:** "Knowledge Graph Construction" (medium.com/@dair.ai)
3. **NetworkX Documentation** - Graph algorithms

**Estimated Time:** 10-12 hours

#### 3.2.4 Multi-Model Ensembles (3-5 days if unfamiliar)

**Core Concepts:**

- Voting strategies (hard, soft, weighted)
- Stacking and blending
- Confidence calibration

**Resources:**

1. **scikit-learn Ensemble Methods** - Documentation
2. **Paper:** "Self-Consistency Improves Chain of Thought" (Wang et al., 2023) - already cited!

**Estimated Time:** 6-8 hours

#### 3.2.5 LLM API Usage & Prompt Engineering (3-5 days)

**Core Concepts:**

- API calls (OpenAI, Anthropic, Google)
- Temperature and sampling parameters
- Chain-of-thought prompting
- Rate limiting and cost management

**Resources:**

1. **OpenAI Cookbook** - GitHub repository
2. **Anthropic Prompt Engineering Guide**
3. **LangChain Documentation** (if using their framework)

**Practical:**

```python
# You'll need to:
import google.generativeai as genai
# Implement robust API calling with retries
# Parse structured outputs
# Manage API costs (3,600 reports × 3 temps = 10,800 API calls!)
```

**Estimated Time:** 6-8 hours

#### 3.2.6 Domain-Specific Knowledge (Ongoing)

**Aviation Safety & ADREP Taxonomy:**

- ICAO ADREP's 11 categories (CFIT, LOC-I, USOS, etc.)
- Aviation terminology and acronyms
- Incident reporting standards

**Resources:**

1. **ICAO ADREP Manual** - Official documentation
2. **FAA Accident Classification** - Taxonomy guide
3. Read 50-100 actual incident reports to internalize the domain

**Estimated Time:** 8-10 hours (reading + note-taking)

### 3.3 Total Learning Time Estimate

| Area                      | Hours                                  |
| ------------------------- | -------------------------------------- |
| Graph Neural Networks     | 20-25                                  |
| NER & Relation Extraction | 15-20                                  |
| Causal Reasoning          | 10-12                                  |
| Ensembles                 | 6-8                                    |
| LLM APIs                  | 6-8                                    |
| Aviation Domain           | 8-10                                   |
| **TOTAL**                 | **65-83 hours** (~2-3 weeks full-time) |

**This is why scope reduction is critical** - you can't spend 3 weeks just learning before coding starts!

---

## Part 4: Technology Stack Breakdown

### 4.1 Core Technologies (Must Use)

| Technology                      | Purpose                 | Learning Curve                | Alternatives       |
| ------------------------------- | ----------------------- | ----------------------------- | ------------------ |
| **PyTorch**                     | Deep learning framework | Medium (you should know this) | TensorFlow/JAX     |
| **PyTorch Geometric**           | GNN library             | Medium-High                   | DGL, GraphNets     |
| **Transformers (Hugging Face)** | BERT/RoBERTa models     | Low-Medium (standard now)     | SpaCy-Transformers |
| **NetworkX**                    | Graph manipulation      | Low                           | igraph             |
| **Pandas/NumPy**                | Data processing         | Low (you should know this)    | Polars             |
| **scikit-learn**                | Metrics & baselines     | Low (you should know this)    | -                  |

### 4.2 Optional/Deferrable Technologies

| Technology                | Purpose           | Defer Until | Why Defer                           |
| ------------------------- | ----------------- | ----------- | ----------------------------------- |
| **Neo4j**                 | Graph database    | Phase 2     | NetworkX is sufficient for research |
| **FastAPI**               | REST API          | Phase 2     | Scripts work for evaluation         |
| **Django**                | Web dashboard     | Phase 2+    | Jupyter/Streamlit is faster         |
| **LangGraph**             | LLM orchestration | Optional    | Direct API calls simpler            |
| **SpaCy (full pipeline)** | NLP preprocessing | Optional    | Hugging Face may suffice            |

### 4.3 Recommended Minimal Stack

For a **feasible semester project**, use:

```
TIER 1 (Essential):
- PyTorch + PyTorch Geometric
- Hugging Face Transformers
- NetworkX
- Python standard library

TIER 2 (Highly Recommended):
- Weights & Biases / TensorBoard (experiment tracking)
- Matplotlib/Seaborn (visualization)
- Jupyter Notebooks (prototyping)

TIER 3 (Nice to Have):
- Streamlit (simple demo UI)
- pytest (unit testing)
```

---

## Part 5: Recommended Revisions

### 5.1 Scope Modifications

**Keep:**

- ✅ Core GNN-based causal reasoning
- ✅ Entity extraction (simplified)
- ✅ Comparison with LLM baseline
- ✅ Explainability focus

**Simplify:**

- ⚠️ Use rule-based relation extraction instead of learned
- ⚠️ Use GCN instead of HGT
- ⚠️ Reduce LLM ensemble to 1-2 models
- ⚠️ Focus on 5-6 ADREP classes instead of all 11

**Cut:**

- ❌ Production deployment (FastAPI, Django)
- ❌ Neo4j database
- ❌ Real-time processing requirements
- ❌ 68K+ report processing (use 3K-5K subset)

### 5.2 Evaluation Redesign

**Replace:**

```
OLD: "supervised accuracy with GPT-4o silver labels"
NEW:
1. Get 200-500 expert-labeled reports (gold standard)
2. Use silver labels only for training
3. Report accuracy on gold holdout set
4. If expert labels unavailable:
   - Multi-rater agreement (Fleiss' Kappa)
   - Manual inspection of 100 cases
   - Error analysis by incident type
```

**Add:**

```
Ablation Studies:
- LLM-only baseline
- Graph-only model
- Combined system
- Report delta improvements

Explainability Evaluation:
- Human assessment of graph quality (N=50)
- Attention weight analysis
- Case studies of corrected "OTHER" classifications
```

### 5.3 Timeline Restructuring

**Proposed 12-Week Schedule:**

| Week | Milestone                      | Deliverables                                 |
| ---- | ------------------------------ | -------------------------------------------- |
| 1-2  | Data prep & EDA                | Cleaned dataset, class distribution analysis |
| 3-4  | GNN learning & implementation  | Working GCN on synthetic graphs              |
| 4-5  | Entity extraction pipeline     | NER output on 100 reports                    |
| 5-6  | Causal graph construction      | Graph generation from reports                |
| 7-8  | GNN training & tuning          | Trained model, initial results               |
| 9    | Baseline comparison            | LLM vs Graph vs Ensemble results             |
| 10   | Ablation studies               | Contribution analysis                        |
| 11   | Error analysis & visualization | Confusion matrices, case studies             |
| 12   | Report writing & presentation  | Final deliverables                           |

**Checkpoints:**

- Week 4: Must have working GNN on toy data
- Week 6: Must have graphs for 500+ reports
- Week 8: Must have trained model (even if accuracy is low)
- Week 10: Must have ablation results

---

## Part 6: Grading Rubric (What I'd Evaluate As TA)

If this were submitted as a final project, here's how I'd grade:

| Criterion                | Weight | What I'm Looking For                                                       |
| ------------------------ | ------ | -------------------------------------------------------------------------- |
| **Technical Soundness**  | 30%    | Correct GNN implementation, valid evaluation methodology, proper baselines |
| **Novelty & Impact**     | 20%    | Clear improvement over baseline, addressing real gap                       |
| **Experimental Rigor**   | 25%    | Ablation studies, statistical significance, error analysis                 |
| **Presentation**         | 15%    | Clear writing, good visualizations, reproducible code                      |
| **Ambition & Execution** | 10%    | Difficulty of problem vs quality of solution                               |

**Current Proposal Score: B- (78/100)**

**Why not higher:**

- Circular evaluation (−10 points)
- Overly ambitious scope (−5 points)
- Missing ablation plan (−5 points)
- Unclear causal extraction (−2 points)

**With revisions: Could be A- or A (90+)**

---

## Part 7: Final Recommendations

### 7.1 For Project Success

1. **Week 1 Action Items:**
   - [ ] Reduce scope using Option A framework above
   - [ ] Acquire 200-500 expert-labeled reports OR plan multi-rater evaluation
   - [ ] Complete GNN tutorials (PyG, CS224W lectures 1-4)
   - [ ] Set up experiment tracking (Weights & Biases)

2. **Technical Priorities:**
   - Start with **simplest possible graph** (just entity co-occurrence)
   - Implement **GCN baseline** before GAT/HGT
   - Get **end-to-end pipeline working** by Week 6 (even if accuracy is low)
   - **Don't prematurely optimize** - research prototype first, production later

3. **Team Composition (if group project):**
   - **Person A:** GNN expert (learns PyG deeply)
   - **Person B:** NLP expert (handles BERT-NER, preprocessing)
   - **Person C:** LLM/evaluation expert (baselines, metrics, analysis)

### 7.2 For Educational Value

**This project teaches:**

- ✅ Graph neural networks (excellent for DL course)
- ✅ Multi-model ensembles
- ✅ Addressing real-world ambiguity (not toy datasets)
- ✅ Explainability in high-stakes domains

**It's a GREAT project conceptually** - just needs scope management!

### 7.3 Risk Mitigation

**Biggest Risks:**

| Risk                         | Probability   | Impact | Mitigation                                |
| ---------------------------- | ------------- | ------ | ----------------------------------------- |
| Can't get gold labels        | High          | High   | Plan multi-rater evaluation now           |
| GNN doesn't improve over LLM | Medium        | High   | Ensure you have good baselines to compare |
| Causal extraction fails      | Medium        | Medium | Have rule-based fallback                  |
| Running out of time          | **Very High** | High   | **Cut scope NOW, not Week 10**            |

---

## Conclusion

**This is a strong proposal with real potential**, but it's **30-40% over-scoped for a semester project**. The core idea—using causal graph reasoning for explainable incident classification—is excellent and publishable if executed well.

**Your immediate priorities:**

1. Get expert labels or redesign evaluation (Week 1)
2. Reduce technology stack to essentials (Week 1)
3. Complete GNN fundamentals (Weeks 1-2)
4. Implement simplest end-to-end pipeline (Weeks 3-4)
5. Iterate and improve (Weeks 5-11)

**Success Criteria for Semester Project:**

- Working GNN-based classifier with 3+ baselines
- Rigorous evaluation (not circular)
- Ablation study showing graph contribution
- 50+ case studies with visualization
- 8-10 page technical report

**If you make these revisions, this could be an A-level project and potentially a workshop paper (e.g., ICML Workshop on AI for Aviation Safety).**

---

## Questions for Office Hours

Before starting, you should be able to answer:

1. How will you extract causal edges (not just entities)?
2. What will you do if you can't get expert labels?
3. How will you isolate the graph contribution from the LLM ensemble?
4. What's your fallback if GNNs don't help?
5. Have you allocated GPU resources (this needs 16GB+ VRAM)?

Feel free to schedule office hours if you want to discuss scope reduction strategies or technical deep-dives on GNNs!

---

**Overall Assessment:**  
**Revise and Resubmit - Strong potential project that needs scope management**

_- Your friendly CMU Deep Learning TA_ 😊
