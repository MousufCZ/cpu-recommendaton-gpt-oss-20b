import pandas as pd
import os
from datetime import datetime

# Paths
RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/raw/cpu_benchmark_v1.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Load CSV 
    df = pd.read_csv(RAW_CSV_PATH)
    
    # --- Save Version 1 (unaltered raw copy) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_v1_path = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v1_{timestamp}.csv")
    df.to_csv(csv_v1_path, index=False)
    print(f"Version 1 saved to: {csv_v1_path}")
    
    # --- Filter only 'Server' category for Version 2 ---
    df_v2 = df[df['category'].str.lower() == 'server']
    
    # --- Save Version 2 --- 
    csv_v2_path = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v2_server_{timestamp}.csv")
    df_v2.to_csv(csv_v2_path, index=False)
    print(f"✅ Version 2 (Server only) saved to: {csv_v2_path}")

    # Show the first 5 rows
    print("First 5 rows of the dataset:\n")
    print(df.head())

    # --- Check data volume ---
    print(f"Number of rows: {df_v2.shape[0]}")
    print(f"Number of columns: {df_v2.shape[1]}")

    # --- Peek at dataset info ---
    print("\nDataset info:")
    print(df_v2.info())  # Shows column names, non-null counts, data types

    # --- Optional: Quick descriptive stats ---
    print("\nSummary statistics (numeric columns):")
    print(df_v2.describe())

if __name__ == "__main__":
    main()
