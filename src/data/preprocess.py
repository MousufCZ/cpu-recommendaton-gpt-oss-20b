import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed/cpu_benchmarks_v2_server_2025-09-08_17-34-24.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Load CSV 
    df = pd.read_csv(RAW_CSV_PATH)
    
    # --- Convert test_date --- 
    df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce').dt.year

    # --- Replace int/float to NaN and Sting/object to Unknown ---
    # Define placeholder values to treat as missing
    placeholders = ["", "N/A", "-", "unknown"]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].replace(placeholders, "Unknown")

    # Save back to the same file (overwrite)
    # df.to_csv(RAW_CSV_PATH, index=False)

    # --- Save Version 3 (cleaned) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_v3_path = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v3_cleaned_{timestamp}.csv")
    df.to_csv(csv_v3_path, index=False)
    print(f"Version 3 (cleaned) saved to: {csv_v3_path}")

    print(f"============================")
    print(f"============================")
    # --- Preview first 10 rows ---
    print(f"Head 10: {df.head(10)}")

    # --- Check data volume ---
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # --- Peek at dataset info ---
    print("\nDataset info:")
    print(df.info())  # Shows column names, non-null counts, data types

    # --- Optional: Quick descriptive stats ---
    print("\nSummary statistics (numeric columns):")
    print(df.describe())

if __name__ == "__main__":
    main()
