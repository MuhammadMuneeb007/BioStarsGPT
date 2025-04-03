import pandas as pd
import os

def filter_questions(input_file, output_file, min_votes=3, min_replies=1):
     
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(input_file, low_memory=False)
    
    # Get the initial count of questions
    initial_count = len(df)
    print(f"Initial number of questions: {initial_count}")
    
    # Print column names to verify
    print("Columns in the dataframe:", df.columns.tolist())
    
    # Convert 'Votes' and 'replies' columns to numeric, coercing errors to NaN
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df['replies'] = pd.to_numeric(df['Replies'], errors='coerce')
    
    # Replace NaN values with 0
    df['Votes'].fillna(0, inplace=True)
    df['replies'].fillna(0, inplace=True)
    
    # Filter based on votes and replies
    filtered_df = df[(df['Votes'] >= min_votes) & (df['replies'] >= min_replies)]
    
    # Get the count of filtered questions
    filtered_count = len(filtered_df)
    print(f"Number of questions after filtering: {filtered_count}")
    print(f"Removed {initial_count - filtered_count} questions")
    
    # Save the filtered questions
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered questions saved to {output_file}")
    
    return filtered_df

if __name__ == "__main__":
    # Define input and output file paths
    input_file = os.path.join(os.path.dirname(__file__), "biostars_all_questions.csv")
    output_file = os.path.join(os.path.dirname(__file__), "biostars_filtered_questions.csv")
    
    # Set filtering criteria
    min_votes = 0
    min_replies = 1
    
    try:
        # Filter questions
        filtered_questions = filter_questions(input_file, output_file, min_votes, min_replies)
        
        # Display sample of filtered questions
        print("\nSample of filtered questions:")
        print(filtered_questions.head())
    except Exception as e:
        print(f"An error occurred: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
