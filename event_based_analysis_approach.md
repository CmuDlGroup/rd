Possible approach

Right now, our models read the *entire narrative* and jumps straight to an ADREP category. 

The idea is to first extract the key events from each report, then classify based on those events rather than raw text.

**Step 1: Extract structured events from the narrative (using transformers)**
From each report, automatically extract:

* **Actors** (pilot, ATC, ground crew, UAS operator)
* **Systems** (engine, hydraulics, navigation, runway, etc.)
* **Phase of flight** (taxi, takeoff, approach, landing…)
* **Causal triggers** (weather, human error, system failure)
* **Outcomes** (incident, diversion, accident, damage)

This uses span-based transformer models (similar to NER, but domain-specific).

**Step 2: Build an event graph**
Turn those extracted elements into a simple causal graph, e.g.:

* *Weather → caused → Hydraulic failure → during → Approach → led to → Diversion*

This mirrors how safety analysts reason about incidents.

**Step 3: Classify using the event graph (not raw text)**
Use a Graph Neural Network (GNN) to classify the *event structure* into ADREP categories, instead of relying only on wording.

**Why this helps**

* Much more **explainable** (we can show *why* a category was chosen)
* More **robust to wording differences**
* Reduces overuse of **OTHER**
* Aligns better with how ICAO/analysts think about incidents
* Fits naturally as an additional model in our existing consensus engine

**Big picture**
We move from:

> “Read text → guess category”

to:

> “Understand what happened → classify based on causal structure”

This keeps our current system intact but adds a more human-like, safety-aligned reasoning layer on top.

**Evaluation**
Evaluation approach: We wouldn’t just measure raw classification accuracy. We’d evaluate the system in layers, the same way safety analysts think. First, we assess event extraction quality (actors, systems, phase of flight, triggers, outcomes) using precision/recall/F1 on a small expert-annotated test set. Second, we compare classification performance across models (text-only, event-only, and event-graph + GNN), focusing on macro-F1 and reductions in structurally impossible errors rather than just top-1 accuracy. Third, we evaluate system-level impact, including reduction in the OTHER category, improvements in consensus behavior (e.g., breaking LLM deadlocks or correctly escalating cases for review), and robustness to wording changes or missing information. Finally, we assess human alignment and explainability by checking whether event-based explanations are clearer and more trustworthy to analysts and whether removing key event nodes meaningfully changes predictions. The goal is to show safer, more robust, and more explainable decisions — not just higher accuracy.