import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

FEATURED_CSV = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/featured/cpu_benchmarks_v5_corrected_testdate_2025-09-08_23-34-43.csv"
ML_DIR = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/ml_ready"
os.makedirs(ML_DIR, exist_ok=True)

def main():
    df = pd.read_csv(FEATURED_CSV)

    print("======================")
    print("Head 10 rows of raw data:")
    print(df.head(10))

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nDataset info:")
    print(df.info())
    print("\nSummary statistics (numeric columns):")
    print(df.describe())
    print("======================")

    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    scaling_map = {
        "price": "RobustScaler",
        "TDP": "RobustScaler",
        "cores": "RobustScaler",
        "price_per_core": "RobustScaler",
        "threadMark_per_watt": "RobustScaler",

        "cpu_mark": "StandardScaler",
        "cpu_value": "StandardScaler",
        "thread_mark": "StandardScaler",
        "thread_value": "StandardScaler",
        "power_performance": "StandardScaler",
        "thread_mark_per_dollar": "StandardScaler",
        "thermal_performance_ratio": "StandardScaler",

        "thread_efficiency": "MinMaxScaler",
        "age": "MinMaxScaler"
    }

    df_scaled = df.copy()
    for col, scaler_name in scaling_map.items():
        if col in df.columns:
            scaler = scalers[scaler_name]
            df_scaled[[col]] = scaler.fit_transform(df[[col]])
            print(f"Applied {scaler_name} to {col}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ML_CSV = os.path.join(ML_DIR, f"cpu_benchmarks_v4_ml_ready_scaled_custom_{timestamp}.csv")
    df_scaled.to_csv(ML_CSV, index=False)
    print(f"\nVersion 4 (ML-ready scaled custom) saved to: {ML_CSV}")


    print("======================")
    print("\nHead 10 rows of scaled data:")
    print(df_scaled.head(10))

if __name__ == "__main__":
    main()
