import pandas as pd
import os

directories = ['outputs_llama3.3', 'outputs_qwen', 'outputs_phi', 'outputs_gemma']
results = []

for directory in directories:
    csv_path = os.path.join(directory, 'detailed_metrics_all.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            last_row = df.tail(1).copy()
            last_row['model Name'] = directory
            results.append(last_row)
        else:
            print(f"{csv_path} is empty.")
    else:
        print(f"{csv_path} does not exist.")

if results:
    final_df = pd.concat(results, ignore_index=True)
    print(final_df.to_markdown())
else:
    print("No results to display.")