import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Set how many entries to process (e.g., 100, 1000, or 5000)
MAX_ENTRIES = 10000

def check_processed_md():
    questions_dir = Path("Questions")
    return [d for d in questions_dir.iterdir() if (d / "text_similarity_analysis.csv").exists()]

def read_csv_files(dirs):
    dataframes = []
    for directory in tqdm(dirs[:MAX_ENTRIES], desc=f"Reading up to {MAX_ENTRIES} CSV files"):
        csv_file = directory / "text_similarity_analysis.csv"
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    return dataframes

def build_aggregated_dataframe(dataframes):
    all_data = pd.concat(dataframes, ignore_index=True)
    
    numeric_columns = all_data.select_dtypes(include=[np.number]).columns.tolist()

    exp1_cols = [col for col in numeric_columns if "_exp1" in col]
    exp2_cols = [col for col in numeric_columns if "_exp2" in col]

    exp1_avg = all_data[exp1_cols].mean()
    exp2_avg = all_data[exp2_cols].mean()

    exp1_avg.index = [col.replace("_exp1", "") for col in exp1_avg.index]
    exp2_avg.index = [col.replace("_exp2", "") for col in exp2_avg.index]

    merged_df = pd.DataFrame([exp1_avg, exp2_avg], index=["Explanation 1", "Explanation 2"])
    return merged_df

def plot_bar_chart(df):
    labels = df.columns.tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df.loc["Explanation 1"], width, label="Explanation 1")
    ax.bar(x + width/2, df.loc["Explanation 2"], width, label="Explanation 2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Average Metric Value")
    ax.set_title("Average Metric Comparison (Bar Chart)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("bar_chart.png", dpi=300)

def main():
    dirs = check_processed_md()
    if not dirs:
        print("No valid directories found.")
        return

    dataframes = read_csv_files(dirs)
    if not dataframes:
        print("No dataframes read.")
        return

    df_avg = build_aggregated_dataframe(dataframes)
    print("\n==== Aggregated Averages Table ====\n")
    print(df_avg.round(3))
    print(df_avg.to_markdown())

if __name__ == "__main__":
    main()
