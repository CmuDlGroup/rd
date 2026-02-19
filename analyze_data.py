
import pandas as pd
import json

DATA_FILE = 'data_with_events_roberta.csv'
TAXONOMY_FILE = 'icao_taxonomy.json'

def analyze():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Data shape: {df.shape}")
        
        print("\nDistribution of 'final_category':")
        print(df['final_category'].value_counts())

        print("\nLoading taxonomy...")
        with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        
        cats = taxonomy.get('occurrence_categories', {})
        print(f"\nNumber of occurrence categories: {len(cats)}")
        print("Sample Categories:")
        for code, details in list(cats.items())[:5]:
            print(f" - {code}: {details.get('name')}")

        print("\nChecking extracted events sample:")
        for i in range(min(5, len(df))):
            try:
                events = json.loads(df.loc[i, 'extracted_events'])
                print(f"Row {i}: {events}")
            except Exception as e:
                print(f"Row {i} error: {e}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze()
