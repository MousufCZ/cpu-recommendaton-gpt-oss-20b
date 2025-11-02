import pandas as pd
import numpy as np
import os
from datetime import datetime

RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed/cpu_benchmarks_v2_server_2025-09-08_17-34-24.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    df = pd.read_csv(RAW_CSV_PATH)
    
    df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce').dt.year

    current_year = datetime.now().year
    df['age'] = current_year - df['test_date']
    
    placeholders = ["", "N/A", "-", "unknown"]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].replace(placeholders, "Unknown")

    # df.to_csv(RAW_CSV_PATH, index=False)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_version_path = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v3_cleaned_{timestamp}.csv")
    df.to_csv(csv_version_path, index=False)
    print(f"Version 3 (cleaned) saved to: {csv_version_path}")

    print(f"============================")
    print(f"============================")

    print(f"Head 10: {df.head(10)}")


    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")


    print("\nDataset info:")
    print(df.info()) 

    print("\nSummary statistics (numeric columns):")
    print(df.describe())

if __name__ == "__main__":
    main()
