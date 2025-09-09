import pandas as pd
import os
from datetime import datetime

# Paths
V1_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/processed/cpu_benchmarks_v1_2025-09-08_17-34-24.csv"  # Original CSV
LATEST_CSV_PATH = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/featured/cpu_benchmarks_v4_feature_engineering_2025-09-08_21-41-12.csv"
PROCESSED_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/featured"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Load original CSV (version 1)
    df_v1 = pd.read_csv(V1_CSV_PATH)
    
    # Load latest feature-engineered CSV
    df_latest = pd.read_csv(LATEST_CSV_PATH)
    
    # Reconstruct full CPU name in latest CSV
    df_latest['cpu_name_full'] = df_latest['brand_name'] + " " + df_latest['cpu_model']
    
    # Merge to get correct test_date from version 1
    df_merged = df_latest.merge(
    df_v1[['cpu_name', 'test_date']],  # Original CPU name and test date
    how='left',
    left_on='cpu_name_full',
    right_on='cpu_name'
    )
    
    # Check columns after merge
    print("Columns after merge:", df_merged.columns)

    # The test_date from df_v1 may now be 'test_date_y'
    if 'test_date_y' in df_merged.columns:
        df_merged['test_date'] = df_merged['test_date_y']

    # Drop temporary columns
    df_merged.drop(columns=['cpu_name_full', 'cpu_name', 'test_date_y'] if 'test_date_y' in df_merged.columns else ['cpu_name_full', 'cpu_name'], inplace=True)
    
    # Recompute age
    current_year = datetime.now().year
    df_merged['age'] = current_year - df_merged['test_date']
    
    # --- Save CSV with corrected test_date and timestamp ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    NEW_CSV = os.path.join(PROCESSED_DIR, f"cpu_benchmarks_v5_corrected_testdate_{timestamp}.csv")
    df_merged.to_csv(NEW_CSV, index=False)
    print(f"\nVersion 5 (corrected test_date) saved to: {NEW_CSV}")

    # Preview
    print(df_merged[['brand_name', 'cpu_model', 'test_date', 'age']].head(10))

if __name__ == "__main__":
    main()
