
import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:toormaster@localhost:5432/aviation"

def verify_db():
    try:
        engine = create_engine(DB_URL)
        print("Connecting to database...")
        
        # Check tables
        original = pd.read_sql("SELECT count(*) as count FROM original_data", engine)
        results = pd.read_sql("SELECT count(*) as count FROM classification_results", engine)
        
        print(f"Rows in 'original_data': {original['count'][0]}")
        print(f"Rows in 'classification_results': {results['count'][0]}")
        
        # Check a sample
        sample = pd.read_sql("SELECT * FROM classification_results LIMIT 3", engine)
        print("\nSample Results:")
        print(sample[['original_category', 'predicted_code', 'method']])
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify_db()
