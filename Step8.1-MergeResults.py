import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to Evaluations directory
evaluations_dir = os.path.join(os.path.dirname(__file__), 'Evaluations')

# Create an empty list to store DataFrames
all_metrics_dfs = []

# Walk through all directories in Evaluations
for root, dirs, files in os.walk(evaluations_dir):
    # Find all metrics.csv files in the current directory
    metrics_files = glob.glob(os.path.join(root, '*metrics.csv'))
    
    for metrics_file in metrics_files:
        try:
            # Read the metrics file into a DataFrame
            df = pd.read_csv(metrics_file)
            
            # Add a column with the directory name for identification
            dir_name = os.path.basename(os.path.dirname(metrics_file))
            df['source_directory'] = dir_name
            #print(f"Processing: {metrics_file} from {dir_name}")
            # Add to the list of DataFrames
            all_metrics_dfs.append(df)
            print(f"Processed: {metrics_file}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

# Check if any metrics files were found
if all_metrics_dfs:
    # Merge all DataFrames
    merged_df = pd.concat(all_metrics_dfs, ignore_index=True)
    
    # Print the original merged DataFrame
    print("\nMerged Metrics Data (Original):")
    print(merged_df)
    
    # First, save the original merged data
    output_path = os.path.join("", 'all_metrics_merged.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")
    
    # Check if 'metric' exists in the DataFrame
    if 'metric' in merged_df.columns:
        # Create a pivot table with metrics as index, source_directory as columns
        pivoted_df = merged_df.pivot_table(
            index='metric',
            columns='source_directory',
            values='score',  # Assuming there's a score column
            aggfunc='mean'   # If there are multiple scores for the same metric/model
        )
        
        # Fill NaN values with 0 or another appropriate placeholder
        pivoted_df = pivoted_df.fillna(0)
        
        # Save the transposed data
        transposed_output_path = os.path.join("", 'transposed_metrics.csv')
        pivoted_df.to_csv(transposed_output_path)
        print(f"\nTransposed data saved to: {transposed_output_path}")
        
        # Create a heatmap
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap
        sns.heatmap(pivoted_df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        
        plt.title('Model Performance Metrics Heatmap')
        plt.tight_layout()
        
        # Save the heatmap as PNG
        heatmap_path = os.path.join("", 'metrics_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nHeatmap saved to: {heatmap_path}")
    else:
        print("\nWarning: 'metric' column not found. Creating transposed view from available metrics columns.")
        
        # Get all metric columns (excluding source_directory which is the model identifier)
        metric_columns = [col for col in merged_df.columns if col != 'source_directory']
        
        # Create a new DataFrame to hold the transposed data
        transposed_data = []
        
        # Group by source_directory and calculate mean for each metric
        grouped = merged_df.groupby('source_directory')
        
        for model, group in grouped:
            # Calculate the mean for each metric column
            row_data = {'metric': model}
            for metric in metric_columns:
                row_data[metric] = group[metric].mean()
            transposed_data.append(row_data)
        
        transposed_df = pd.DataFrame(transposed_data)
        transposed_df.set_index('metric', inplace=True)
        
        # Now pivot this dataframe to have models as columns and metrics as rows
        final_df = transposed_df.T  # Transpose to get metrics as rows and models as columns
        
        # Save the transposed data
        transposed_output_path = os.path.join("", 'transposed_metrics.csv')
        final_df.to_csv(transposed_output_path)
        print(f"\nTransposed data saved to: {transposed_output_path}")
        
        # Create a heatmap
        plt.figure(figsize=(16, 12))
        
        # Create the heatmap with scaled colors
        sns.heatmap(final_df, annot=True, cmap='viridis', fmt=".3f", linewidths=.5, 
                   annot_kws={"size": 8})
        
        plt.title('Model Performance Metrics Heatmap')
        plt.tight_layout()
        
        # Save the heatmap as PNG
        heatmap_path = os.path.join("", 'metrics_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nHeatmap saved to: {heatmap_path}")
        
        # Create individual heatmaps for better readability
        # Split metrics into groups of 5 for better visualization
        chunk_size = 5
        for i in range(0, len(metric_columns), chunk_size):
            chunk_metrics = metric_columns[i:i+chunk_size]
            plt.figure(figsize=(14, 8))
            
            # Select only the current chunk of metrics
            chunk_df = final_df.loc[chunk_metrics]
            
            # Create the heatmap with scaled colors
            sns.heatmap(chunk_df, annot=True, cmap='viridis', fmt=".3f", linewidths=.5,
                       annot_kws={"size": 10})
            
            plt.title(f'Model Performance Metrics Heatmap (Group {i//chunk_size + 1})')
            plt.tight_layout()
            
            # Save the chunked heatmap as PNG
            chunked_heatmap_path = os.path.join("", f'metrics_heatmap_group_{i//chunk_size + 1}.png')
            plt.savefig(chunked_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nChunked heatmap saved to: {chunked_heatmap_path}")
else:
    print("No metrics.csv files found in the Evaluations directory.")
