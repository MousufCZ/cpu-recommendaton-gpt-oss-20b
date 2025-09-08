import pandas as pd
import os
from datetime import datetime

# Paths
RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/raw/cpu_benchmark_v1.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Load CSV into a DataFrame
    df = pd.read_csv(RAW_CSV_PATH)
    
    # Show the first 5 rows
    print("First 5 rows of the dataset:\n")
    print(df.head())

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
