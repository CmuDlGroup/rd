
import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:toormaster@localhost:5432/aviation"

def verify_final():
    try:
        engine = create_engine(DB_URL)
        print("Connecting to database...")
        
        # 1. Check Row Counts
        orig_count = pd.read_sql("SELECT count(*) as c FROM original_data", engine)['c'][0]
        res_count = pd.read_sql("SELECT count(*) as c FROM classification_results", engine)['c'][0]
        
        print(f"Original Rows: {orig_count}")
        print(f"Result Rows: {res_count}")
        
        if orig_count != res_count:
            print("WARNING: Row count mismatch!")
        else:
            print("Row counts match.")
            
        # 2. Check Event ID Linkage
        print("\nChecking Event ID Linkage (Sample Join)...")
        query = """
        SELECT 
            o.event_id, 
            o.final_category as orig_cat, 
            r.predicted_code, 
            r.confidence 
        FROM original_data o
        JOIN classification_results r ON o.event_id = r.event_id
        LIMIT 5;
        """
        
        joined_df = pd.read_sql(query, engine)
        if joined_df.empty:
            print("ERROR: Join failed! IDs might not match.")
        else:
            print("Join successful. Sample:")
            print(joined_df)
            
        # 3. Check for Duplicates in Results
        dup_query = "SELECT event_id, count(*) FROM classification_results GROUP BY event_id HAVING count(*) > 1"
        dups = pd.read_sql(dup_query, engine)
        if not dups.empty:
            print(f"ERROR: Found {len(dups)} duplicate event_ids in results!")
        else:
            print("No duplicates found in classification_results.")

    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify_final()
