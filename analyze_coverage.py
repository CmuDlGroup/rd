import pandas as pd
import json

df = pd.read_csv('data_with_events.csv')
empty_kw_count = 0
empty_both_count = 0

print(f"Total rows: {len(df)}")

for index, row in df.iterrows():
    try:
        data = json.loads(row['extracted_events'])
        kw = data.get('keyword_extraction', {})
        qa = data.get('qa_extraction', {})
        
        kw_empty = all(len(v) == 0 for v in kw.values())
        qa_empty = all(v is None for v in qa.values())
        
        if kw_empty:
            empty_kw_count += 1
            if qa_empty:
                empty_both_count += 1
                if empty_both_count <= 3:
                     print(f"\n--- Row {index} (Both Empty) ---\nNarrative: {row.get('narrative_1')[:100]}...")
            elif empty_kw_count <= 3:
                print(f"\n--- Row {index} (KW Empty, QA Found) ---\nNarrative: {row.get('narrative_1')[:100]}...")
                print(f"QA: {qa}")

    except Exception as e:
        print(f"Error parsing row {index}: {e}")

print(f"\nRows with empty Keyword Extraction: {empty_kw_count}")
print(f"Rows with both Empty: {empty_both_count}")
