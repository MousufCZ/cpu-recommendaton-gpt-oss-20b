import pandas as pd
import os
from datetime import datetime

# Paths
RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed/cpu_benchmarks_v2_server_2025-09-08_17-34-24.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Load CSV 
    df = pd.read_csv(RAW_CSV_PATH)
    
    # # --- Split cpu_name into brand_name and cpu_model ---
    # df["brand_name"] = df["cpu_name"].str.split(" ", n=1).str[0]  # first word
    # df["cpu_model"] = df["cpu_name"].str.split(" ", n=1).str[1]   # everything else

    # # Drop original column
    # df.drop(columns=["cpu_name"], inplace=True)

    # Reorder columns: brand_name, cpu_model first, then the rest
    # cols = ["brand_name", "cpu_model"] + [col for col in df.columns if col not in ["brand_name", "cpu_model"]]
    # df = df[cols]

    df = df.sort_values(by=["brand_name", "test_date"], ascending=[True, False])

    # Save back to the same file (overwrite)
    df.to_csv(RAW_CSV_PATH, index=False)

    print(f"✅ Updated file saved (cpu_name removed, brand_name + cpu_model added).")
    print(df.head(10))  # preview first 10 rows

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
