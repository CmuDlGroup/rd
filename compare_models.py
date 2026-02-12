import pandas as pd
import json
import time
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm
import collections

# --- Configuration ---
INPUT_FILE = 'data-1770316648579.csv'
OUTPUT_FILE = 'comparison_results.csv'
METRICS_FILE = 'model_metrics.md'

MODELS = {
    "distilbert": "distilbert-base-cased-distilled-squad",
    "roberta": "deepset/roberta-base-squad2",
    "longformer": "valhalla/longformer-base-4096-finetuned-squad",
    "aerobert": "NASA-AIML/MIKA_SafeAeroBERT"
}

QUESTIONS = {
    "ACTOR": "Who was the primary actor involved?",
    "SYSTEM": "What aircraft system or component failed or was involved?",
    "PHASE": "What phase of flight was the aircraft in?",
    "TRIGGER": "What caused the event or incident?",
    "OUTCOME": "What was the final outcome or result of the event?"
}

def load_pipeline(model_name):
    print(f"Loading {model_name}...")
    try:
        # Note: pipeline handles model instantiation. using handle_impossible_answer for robust models
        # For MIKA (AeroBERT), it might not have a QA head, so we expect warnings.
        nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
        return nlp
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings (token overlap)."""
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import string, re
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, F1 is 1 if both are empty, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main():
    df = pd.read_csv(INPUT_FILE)
    # Using a subset for faster iteration during dev/test if needed, usually full set
    # df = df.head(10) 

    # Load pipelines
    pipelines = {}
    for name, path in MODELS.items():
        pipelines[name] = load_pipeline(path)

    results = []
    
    print("Starting comparison...")
    start_time = time.time()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        narrative = str(row.get('narrative_1', ''))
        if not narrative or narrative.lower() == 'nan':
            continue

        row_result = {'row_id': index, 'narrative_snippet': narrative[:50]}
        
        # We will use RoBERTa as the "Reference" for F1 calculation relative to others, 
        # just to show agreement, since we have no gold standard.
        
        extracted_per_model = {} # Store answers for cross-comparison

        for model_name, nlp in pipelines.items():
            if not nlp:
                continue
            
            model_data = {}
            for category, question in QUESTIONS.items():
                try:
                    ans = nlp(question=question, context=narrative)
                    # ans keys: score, start, end, answer
                    model_data[f"{category}_ans"] = ans['answer']
                    model_data[f"{category}_conf"] = ans['score']
                except Exception:
                    model_data[f"{category}_ans"] = ""
                    model_data[f"{category}_conf"] = 0.0
            
            extracted_per_model[model_name] = model_data
            
            # Flatten for CSV
            for k, v in model_data.items():
                row_result[f"{model_name}_{k}"] = v

        # Compute Agreement Metrics (RoBERTa as anchor)
        if 'roberta' in extracted_per_model:
            ref = extracted_per_model['roberta']
            for other_model in ['distilbert', 'longformer', 'aerobert']:
                if other_model in extracted_per_model:
                    comp = extracted_per_model[other_model]
                    # Average F1 across all 5 questions
                    f1_sum = 0
                    for cat in QUESTIONS:
                        f1_sum += compute_f1(ref[f"{cat}_ans"], comp[f"{cat}_ans"])
                    row_result[f"agreement_roberta_{other_model}_f1"] = f1_sum / 5.0
        
        results.append(row_result)

    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    # Generate Summary Report
    avg_conf = {}
    for model in MODELS:
        cols = [c for c in res_df.columns if c.startswith(f"{model}_") and c.endswith("_conf")]
        if cols:
            avg_conf[model] = res_df[cols].mean().mean() # Mean of means
    
    avg_agreement = {}
    for col in res_df.columns:
        if col.startswith("agreement_"):
            avg_agreement[col] = res_df[col].mean()

    with open(METRICS_FILE, 'w') as f:
        f.write("# Model Comparison Metrics\n\n")
        f.write("## Average Confidence Scores (IR Score)\n")
        f.write("| Model | Avg Confidence |\n|---|---|\n")
        for m, s in avg_conf.items():
            f.write(f"| {m} | {s:.4f} |\n")
        
        f.write("\n## Inter-Model Agreement (Relative to RoBERTa)\n")
        f.write("Since we lack ground truth, we measure how much other models agree with RoBERTa (SQuAD2 strong baseline).\n\n")
        f.write("| Comparison | Avg F1 Agreement |\n|---|---|\n")
        for k, v in avg_agreement.items():
            f.write(f"| {k} | {v:.4f} |\n")

    print(f"Results saved to {OUTPUT_FILE} and {METRICS_FILE}.")

if __name__ == "__main__":
    main()
