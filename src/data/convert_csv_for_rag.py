import pandas as pd
import os
from datetime import datetime

INPUT_CSV = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/final/cpu_benchmarks_final.csv"   # your final dataset
OUTPUT_CSV = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/documents/"        # new file with text chunks

def format_value(value, col):
    """Format values based on column rules."""
    if pd.isna(value) or value == "":
        return "Null"

    try:
        num = float(value)
        if col in ["age", "cores", "test_date"] and num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return str(value) if str(value).strip() else "Unknown"

def row_to_text(row):
    """Convert a row of CPU data to a single descriptive text chunk."""
    return (
        f"The brand_name is {format_value(row['brand_name'], 'brand_name')}, "
        f"cpu_model is {format_value(row['cpu_model'], 'cpu_model')} at the price of {format_value(row['price'], 'price')}. "
        f"The cpu_mark is {format_value(row['cpu_mark'], 'cpu_mark')}. "
        f"The cpu_value is {format_value(row['cpu_value'], 'cpu_value')}. "
        f"The thread_mark is {format_value(row['thread_mark'], 'thread_mark')}. "
        f"The thread_value is {format_value(row['thread_value'], 'thread_value')}. "
        f"The TDP is {format_value(row['TDP'], 'TDP')}. "
        f"The power_performance is operating at {format_value(row['power_performance'], 'power_performance')}. "
        f"The number of cores are {format_value(row['cores'], 'cores')}. "
        f"The socket this CPU is suitable for is {format_value(row['socket'], 'socket type')}. "
        f"This cpu is for {format_value(row['category'], 'category')}. "
        f"The price_per_core is {format_value(row['price_per_core'], 'price_per_core')} and the thread_mark_per_dollar is {format_value(row['thread_mark_per_dollar'], 'thread_mark_per_dollar')}. "
        f"The thread_efficiency is {format_value(row['thread_efficiency'], 'thread_efficiency')}. "
        f"The threadMark_per_watt is {format_value(row['threadMark_per_watt'], 'threadMark_per_watt')}. "
        f"The thermal_performance_ratio is {format_value(row['thermal_performance_ratio'], 'thermal_performance_ratio')}. "
        f"The age of the cpu is {format_value(row['age'], 'age')} and it was tested in the year {format_value(row['test_date'], 'test_date')}."
    )

def main():
    # Load dataset
    df = pd.read_csv(INPUT_CSV)

    # Convert rows to text chunks
    text_chunks = df.apply(row_to_text, axis=1).tolist()

    # Save to new CSV with timestamp
    df_text = pd.DataFrame({"cpu_description": text_chunks})
    os.makedirs(OUTPUT_CSV, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = os.path.join(OUTPUT_CSV, f"cpu_text_chunks_{timestamp}.csv")
    df_text.to_csv(output_csv, index=False)

    print(f"\nCPU text chunks saved to: {output_csv}")

if __name__ == "__main__":
    main()
