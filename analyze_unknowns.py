
import pandas as pd

RESULTS_FILE = "classification_results_refined.csv"

def analyze_unknowns():
    try:
        df = pd.read_csv(RESULTS_FILE)
        
        # Filter for Low Confidence / UNK
        unk_df = df[df['predicted_code'] == 'UNK']
        
        print(f"Total processed: {len(df)}")
        print(f"Total UNK: {len(unk_df)}")
        print(f"Percentage UNK: {len(unk_df)/len(df)*100:.1f}%")
        
        print("\n--- Analysis of UNK Rows ---")
        
        # Check for Empty/Short Event Strings
        # We constructed the string as "Phase: ... Actor: ..."
        # A "mostly empty" string would look like "Phase: . Actor: . Trigger: . System: . Outcome: ."
        # Let's count effective length (removing the template words)
        def effective_length(s):
            template_words = ["Phase:", "Actor:", "Trigger:", "System:", "Outcome:"]
            s_clean = s
            for w in template_words:
                s_clean = s_clean.replace(w, "")
            return len(s_clean.strip())

        unk_df['eff_len'] = unk_df['event_string'].apply(effective_length)
        
        short_events = unk_df[unk_df['eff_len'] < 10]
        print(f"\n1. Poor Extraction (Effective Length < 10 chars): {len(short_events)} rows")
        if not short_events.empty:
            print("   Examples:")
            for s in short_events['event_string'].head(3):
                print(f"   - '{s}'")

        print(f"\n2. Sufficient Extraction but Low Match (Effective Length >= 10 chars): {len(unk_df) - len(short_events)} rows")
        good_extraction_unk = unk_df[unk_df['eff_len'] >= 10].sort_values(by='confidence', ascending=False)
        
        if not good_extraction_unk.empty:
            print("   Top High-Confidence 'UNK' (almost made threshold):")
            for _, row in good_extraction_unk.head(5).iterrows():
                print(f"   - Conf: {row['confidence']:.4f} | Event: {row['event_string']}")
            
            print("\n   Lowest Confidence (Complete Mismatch):")
            for _, row in good_extraction_unk.tail(5).iterrows():
                print(f"   - Conf: {row['confidence']:.4f} | Event: {row['event_string']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_unknowns()
