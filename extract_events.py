import pandas as pd
import re
import json
from transformers import pipeline
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = 'data-1770316648579.csv'
OUTPUT_FILE = 'data_with_events_roberta.csv'

# Dictionary for Keyword Extractor
KEYWORD_MAPPINGS = {
    "ACTOR": [
        "Pilot", "Captain", "First Officer", "Co-Pilot", "Crew", "Instructor", "Student", 
        "Controller", "ATC", "Tower", "Ground", "Maintenance", "Mechanic", "UAS Operator"
    ],
    "SYSTEM": [
        "Engine", "Gear", "Landing Gear", "Flaps", "Hydraulics", "Brakes", "Avionics", 
        "Autopilot", "TCAS", "GPWS", "Navigation", "Radio", "Transponder", "UAS", "Drone", "Battery"
    ],
    "PHASE": [
        "Taxi", "Takeoff", "Climb", "Cruise", "Descent", "Approach", "Final Approach", 
        "Landing", "Go-around", "Hover", "Maneuvering"
    ],
    "TRIGGER": [
        "Turbulence", "Wind", "Shear", "Icing", "Strike", "Bird", "Traffic", "Conflict", 
        "Failure", "Malfunction", "Error", "Mistake", "Confusion", "Fatigue", "Distraction"
    ],
    "OUTCOME": [
        "Incident", "Accident", "Collision", "NMAC", "Near Miss", "CFTT", "CFIT", 
        "Runway Incursion", "Excursion", "Injury", "Damage", "Diversion", "Return", "Go-around"
    ]
}

# Questions for QA Extractor
QA_QUESTIONS = {
    "ACTOR": "Who was the primary actor involved?",
    "SYSTEM": "What aircraft system or component failed or was involved?",
    "PHASE": "What phase of flight was the aircraft in?",
    "TRIGGER": "What caused the event or incident?",
    "OUTCOME": "What was the final outcome or result of the event?"
}

class KeywordMatchExtractor:
    def __init__(self, mappings):
        self.mappings = mappings

    def extract(self, text):
        results = {key: [] for key in self.mappings}
        text_lower = text.lower()
        
        for category, keywords in self.mappings.items():
            for keyword in keywords:
                # Simple whole word matching to avoid substring issues (e.g. "age" in "damage")
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    results[category].append(keyword)
        
        # Deduplicate
        for key in results:
            results[key] = list(set(results[key]))
        return results

class TransformerQAExtractor:
    def __init__(self, model_name="deepset/roberta-base-squad2", confidence_threshold=0.1):
        print(f"Loading QA model: {model_name} (Robust SQuAD2)...")
        try:
            self.pipe = pipeline("question-answering", model=model_name)
            self.enabled = True
        except Exception as e:
            print(f"Failed to load QA model: {e}")
            self.enabled = False
        self.questions = QA_QUESTIONS
        self.threshold = confidence_threshold

    def extract(self, text):
        if not self.enabled:
            return {}
        
        results = {}
        for category, question in self.questions.items():
            try:
                # QA pipeline returns {'score': float, 'start': int, 'end': int, 'answer': str}
                answer_data = self.pipe(question=question, context=text)
                if answer_data['score'] >= self.threshold:
                    results[category] = answer_data['answer']
                else:
                    results[category] = None
            except Exception as e:
                # print(f"Error answering {question}: {e}")
                results[category] = None
        return results

def main():
    print(f"Reading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # Initialize Extractors
    keyword_extractor = KeywordMatchExtractor(KEYWORD_MAPPINGS)
    qa_extractor = TransformerQAExtractor() 

    extracted_data = []

    print("Starting extraction...")
    # Using a subset for testing if needed, but processing all for now
    # df = df.head(10) 
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        narrative = str(row.get('narrative_1', ''))
        if not narrative or narrative.lower() == 'nan':
            extracted_data.append({})
            continue

        # 1. Keyword Extraction
        kw_results = keyword_extractor.extract(narrative)

        # 2. QA Extraction
        qa_results = qa_extractor.extract(narrative)

        # Combine results
        # We will store them separately to allow comparison, or merge them.
        # For this task, let's create a structured object containing both.
        
        combined_entry = {
            "keyword_extraction": kw_results,
            "qa_extraction": qa_results
        }
        
        extracted_data.append(combined_entry)

        # Print sample for verification
        if index < 5:
            print(f"\n--- Entry {index} ---")
            print(f"Narrative: {narrative[:200]}...")
            print(f"Keywords: {kw_results}")
            print(f"QA: {qa_results}")

    # Add to DataFrame
    df['extracted_events'] = [json.dumps(d) for d in extracted_data]

    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
