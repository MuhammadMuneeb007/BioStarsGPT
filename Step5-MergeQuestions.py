import json
import os
import csv
import sys
import re
from collections import OrderedDict

def read_json_file(file_path):
    """Read and return the content of a JSON file with preserved field order."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_existing_csv(file_path):
    """Read existing CSV into a dictionary with QuestionCount as key."""
    questions_dict = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Use the QuestionCount as the key
                    questions_dict[row['QuestionCount']] = row
            print(f"Successfully read {len(questions_dict)} existing entries from {file_path}")
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
    return questions_dict

def clean_explanation(text):
    """Remove source links from explanation text."""
    if not text:
        return ""
    
    # Handle the specific URL pattern seen in the data including trailing quotes/slashes
    specific_pattern = r'\n*\n+SOURCE LINK: https://www\.biostars\.org/p/\d+/?"?'
    cleaned_text = re.sub(specific_pattern, '', text, flags=re.IGNORECASE)
    
    # Handle any URL following SOURCE LINK pattern with more flexibility
    general_pattern = r'\n*\n+SOURCE LINK:.*?(https?://\S+).*?($|\n)'
    cleaned_text = re.sub(general_pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # More specific line-by-line filtering
    lines = cleaned_text.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip any line containing SOURCE LINK
        if 'SOURCE LINK' in line.upper():
            continue
        # Skip any line that's just a URL to biostars.org
        if re.search(r'https?://www\.biostars\.org/p/\d+/?', line):
            continue
        filtered_lines.append(line)
    
    cleaned_text = '\n'.join(filtered_lines)
    
    # Remove trailing empty lines
    cleaned_text = cleaned_text.rstrip()
    
    # If the text ends with just a URL, remove that too
    cleaned_text = re.sub(r'\n+https?://\S+$', '', cleaned_text)
    
    return cleaned_text

def fix_and_enrich_json(json_data, csv_entry=None, dir_name=None):
    """Fix field names and enrich JSON with CSV data if available, preserving field order."""
    if not isinstance(json_data, list):
        json_data = [json_data] if json_data else []
    
    # Process each entry in the JSON data
    processed_data = []
    
    for i, entry in enumerate(json_data):
        # Create a new OrderedDict to preserve field order
        processed_entry = OrderedDict()
        
        # First add all original fields in their original order
        for key, value in entry.items():
            # Replace "Explanation" with "Explanation1"
            if key == "Explanation":
                processed_entry["Explanation1"] = clean_explanation(value)
                #print(f"   ✓ Renamed 'Explanation' to 'Explanation1'")
            elif key.startswith("Explanation"):
                # Clean any explanation field (Explanation1, Explanation2, etc.)
                processed_entry[key] = clean_explanation(value)
            else:
                processed_entry[key] = value
        
        # Add QuestionCount field if not present
        if dir_name is not None and "QuestionCount" not in processed_entry:
            processed_entry["QuestionCount"] = dir_name
        
        # If CSV data is available, enrich the entry
        if csv_entry:
            # Process Tags: split by pipe character into separate array elements
            if csv_entry.get('Tags'):
                # Split by pipe OR comma (depending on what's in the data)
                tags_str = csv_entry.get('Tags', "")
                if '|' in tags_str:
                    processed_entry["Tags"] = [tag.strip() for tag in tags_str.split('|')]
                elif ',' in tags_str:
                    processed_entry["Tags"] = [tag.strip() for tag in tags_str.split(',')]
                else:
                    processed_entry["Tags"] = [tags_str] if tags_str else []
            else:
                processed_entry["Tags"] = []
            
            # Add other CSV data
            processed_entry["URL"] = csv_entry.get('URL', "")
            processed_entry["Votes"] = float(csv_entry.get('Votes', 0))
            processed_entry["Views"] = float(csv_entry.get('Views', 0))
            processed_entry["Replies"] = int(csv_entry.get('Replies', 0))
            processed_entry["Title"] = csv_entry.get('Title', "").replace("'","").replace("`","")  # Keep Title as a string
            
            # We're no longer adding source links to explanations
            # Instead, we've already cleaned the explanation fields above
        
        processed_data.append(processed_entry)
    
    return processed_data

def natural_sort_key(s):
    """
    Sort strings containing numbers naturally (e.g., 'file1', 'file2', 'file10' in that order)
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# Main execution
if __name__ == "__main__":
    # CSV file path in current directory
    csv_file_path = "biostars_filtered_questions.csv"
    
    # Read existing CSV data
    csv_data = read_existing_csv(csv_file_path)
    
    # Counter variables
    total_dirs = 0
    dirs_with_json = 0
    dirs_without_json = 0
    processed_files = 0
    renamed_fields = 0
    
    # Get all subdirectories in the Questions directory
    questions_dir = "Questions"
    question_dirs = []
    
    # List to store all processed questions
    all_questions = []
    
    # Check if Questions directory exists
    if os.path.exists(questions_dir) and os.path.isdir(questions_dir):
        # Get all subdirectories in Questions and sort them naturally
        all_items = sorted(os.listdir(questions_dir), key=natural_sort_key)
        
        # Process the sorted items
        for item in all_items:
            subdir_path = os.path.join(questions_dir, item)
            if os.path.isdir(subdir_path):
                total_dirs += 1
                question_file = os.path.join(subdir_path, "Question.json")
                if os.path.exists(question_file):
                    question_dirs.append(subdir_path)
                    dirs_with_json += 1
                else:
                    dirs_without_json += 1
    
    print(f"Total directories: {total_dirs}")
    print(f"Directories with JSON files: {dirs_with_json}")
    print(f"Directories without JSON files: {dirs_without_json}")
    
    # TEST_MODE can be set to False to process all directories
    TEST_MODE = False
    max_dirs_to_process = 50 if TEST_MODE else len(question_dirs)
    processed_dirs = 0
    
    # Process each directory
    for dir_path in question_dirs:
        if processed_dirs >= max_dirs_to_process:
            break
            
        # Get directory name which corresponds to QuestionCount in CSV
        dir_name = os.path.basename(dir_path)
        json_file_path = os.path.join(dir_path, "Question.json")
        
        # Read JSON file regardless of CSV match (we want all questions)
        json_data = read_json_file(json_file_path)
        
        if json_data:
            #print(f"\n==========================================")
            #print(f"PROCESSING: {json_file_path}")
            
            # Check if we have a matching CSV entry for this directory
            if dir_name in csv_data:
                csv_entry = csv_data[dir_name]
                #print(f"Found matching CSV entry with QuestionCount={dir_name}")
                
                # Fix and enrich JSON with CSV data
                processed_json = fix_and_enrich_json(json_data, csv_entry, dir_name)
                all_questions.extend(processed_json)
            else:
                print(f"No matching CSV entry found for directory {dir_name}")
                # Still add the question, but without enrichment
                processed_json = fix_and_enrich_json(json_data, None, dir_name)
                all_questions.extend(processed_json)
            
            processed_files += 1
        else:
            print(f"No JSON data found in {json_file_path}")
        
        processed_dirs += 1
    
    # Write all questions to a single JSON file with preserved field order
    output_file = "Allquestions.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, indent=2)
        print(f"\nSuccessfully wrote {len(all_questions)} questions to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
    
    print("\n==========================================")
    print(f"SUMMARY:")
    print(f"Directories processed: {processed_dirs}")
    print(f"Questions collected: {len(all_questions)}")
    print(f"Output file: {output_file}")
    print("==========================================")