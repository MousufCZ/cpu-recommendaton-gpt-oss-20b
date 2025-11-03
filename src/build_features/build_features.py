import pandas as pd
import numpy as np
import os
from datetime import datetime

RAW_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed/cpu_benchmarks_v3_cleaned_2025-09-08_20-52-44.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    df = pd.read_csv(RAW_CSV_PATH)

    current_year = datetime.now().year

    # --- Derived Features ---
    # Cost-performance
    df['price_per_core'] = df['price'] / df['cores']
    df['thread_mark_per_dollar'] = df['thread_mark'] / df['price']

    # Parallelism
    df['thread_efficiency'] = df['thread_mark'] / df['cpu_mark']

    # Efficiency / Thermals
    df['threadMark_per_watt'] = df['thread_mark'] / df['TDP']
    df['thermal_performance_ratio'] = df['power_performance'] / df['cpu_value']

    # Architecture Evolution / Age
    df['age'] = current_year - df['test_date']

    # df.to_csv(RAW_CSV_PATH, index=False)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_version_path = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v4_feature_engineering_{timestamp}.csv")
    df.to_csv(csv_version_path, index=False)
    print(f"Version 4 (feature engineering) saved to: {csv_version_path}")

    print(f"============================")
    print(f"============================")
    print(f"Head 10: {df.hsead(10)}")

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nDataset info:")
    print(df.info()) 

    print("\nSummary statistics (numeric columns):")
    print(df.describe())

if __name__ == "__main__":
    main()