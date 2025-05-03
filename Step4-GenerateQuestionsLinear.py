import os
import pandas as pd
import re
import json
import time
import threading
import queue
import datetime
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import concurrent.futures

def load_questions_data(csv_path: str) -> pd.DataFrame:
    """
    Load questions data from CSV file and filter out questions that already have valid JSON files
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame containing only questions that need processing
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} questions from CSV")
    
    
    # Get the base directory for questions
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_folder = os.path.join(base_dir, "Questions")
    
    # Create a list to store questions that need processing
    questions_to_process = []
    
    # Check each question
    for _, row in df.iterrows():
        question_id = row['QuestionCount']
        question_dir = os.path.join(question_folder, str(question_id))
        json_path = os.path.join(question_dir, "Question.json")
        
        # Case 1: Directory doesn't exist - need to process
        if not os.path.exists(question_dir):
            questions_to_process.append(row)
            continue
            
        # Case 2: JSON file doesn't exist - need to process
        if not os.path.exists(json_path):
            questions_to_process.append(row)
            continue
            
        # Case 3: JSON file exists but is corrupt - need to process
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                json.loads(json_content)  # Try to parse the JSON
                # JSON is valid, skip this question
        except (json.JSONDecodeError, Exception):
            # JSON is corrupt or can't be read - need to process
            questions_to_process.append(row)
    
    # Create a new DataFrame with only questions that need processing
    filtered_df = pd.DataFrame(questions_to_process)
    print(f"Found {len(filtered_df)} questions that need processing (missing directory, missing JSON, or corrupt JSON)")
    
    return filtered_df
 
 
def read_question_content(question_folder: str, question_id: int) -> Tuple[str, str, List[str]]:
    """
    Read question content from markdown file
    
    Args:
        question_folder: Base folder containing all questions
        question_id: ID of the question to read
        
    Returns:
        Tuple of (title, content, answers)
    """
    markdown_path = os.path.join(question_folder, str(question_id), "Text.md")
    
    if not os.path.exists(markdown_path):
        return "Question not found", "", []
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract title (first line starting with #)
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled Question"
        
        # Extract question content (between metadata and ## Answers or ## Tags)
        content_match = re.search(r'\*\*Posted by:.+?\*\*\s+(.+?)(?=\n## (Answers|Tags)|$)', content, re.DOTALL)
        question_content = content_match.group(1).strip() if content_match else ""
        
        # Extract answers
        answers = []
        answer_sections = re.findall(r'### Answer \d+.*?\n\n(.*?)(?=\n### Answer \d+|\n## Tags|\Z)', content, re.DOTALL)
        for answer in answer_sections:
            answers.append(answer.strip())
        
        return title, question_content, answers
    
    except Exception as e:
        print(f"Error reading question {question_id}: {e}")
        return "Error reading question", "", []

def extract_json_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract valid JSON from model response text with enhanced error handling
    for control characters and common formatting issues
    
    Args:
        response_text: Text response from the model
        
    Returns:
        Parsed JSON as a list of dictionaries
    """
    try:
        # Try to find JSON array within triple backticks
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
            
            # Clean the JSON string to handle common issues
            json_str = clean_json_string(json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error in initial JSON parsing within backticks: {e}")
                # Log the problematic portion
                error_position = e.pos
                context_start = max(0, error_position - 30)
                context_end = min(len(json_str), error_position + 30)
                print(f"Context around error: ...{json_str[context_start:error_position]}[ERROR HERE]{json_str[error_position:context_end]}...")
                
                # Repair specific issues based on error type
                if "Unterminated string" in str(e):
                    print("Detected unterminated string, attempting advanced string repair...")
                    json_str = repair_unterminated_strings_v2(json_str, error_position)
                else:
                    json_str = repair_unterminated_strings(json_str)
                    
                return json.loads(json_str)
        
        # If not found in code block, try cleaning and parsing the entire text
        cleaned_text = clean_json_string(response_text)
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"Error in JSON parsing of cleaned text: {e}")
            error_position = e.pos
            context_start = max(0, error_position - 30)
            context_end = min(len(cleaned_text), error_position + 30)
            print(f"Context around error: ...{cleaned_text[context_start:error_position]}[ERROR HERE]{cleaned_text[error_position:context_end]}...")
            
            # Try to repair unterminated strings
            if "Unterminated string" in str(e):
                repaired_text = repair_unterminated_strings_v2(cleaned_text, error_position)
            else:
                repaired_text = repair_unterminated_strings(cleaned_text)
                
            return json.loads(repaired_text)
            
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        
        # Attempt more aggressive JSON extraction
        try:
            print("Attempting more aggressive JSON extraction...")
            
            # Find anything that looks like a JSON array
            json_array_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if json_array_match:
                potential_json = json_array_match.group(0)
                cleaned_json = clean_json_string(potential_json)
                repaired_json = repair_unterminated_strings(cleaned_json)
                
                try:
                    return json.loads(repaired_json)
                except json.JSONDecodeError as e:
                    print(f"Error in aggressive extraction: {e}")
                    
                    # Extreme approach: extract individual JSON objects and rebuild the array
                    print("Attempting to extract and rebuild individual JSON objects...")
                    json_objects = extract_json_objects_from_text(response_text)
                    if json_objects:
                        return json_objects
            
            # If still unsuccessful, fall back to regex pattern extraction
            print("Falling back to regex pattern extraction...")
            return build_json_from_patterns(response_text)
                
        except Exception as e2:
            print(f"Advanced extraction also failed: {e2}")
            # Final fallback
            return manually_fix_json(response_text)

def repair_unterminated_strings(json_str: str) -> str:
    """
    Repair unterminated strings in JSON
    
    Args:
        json_str: JSON string that might contain unterminated strings
        
    Returns:
        Repaired JSON string
    """
    # Find strings that start with " but don't end with " before the next ,
    lines = json_str.split('\n')
    repaired_lines = []
    
    for line in lines:
        # Check for unterminated strings in the line
        quote_positions = [i for i, char in enumerate(line) if char == '"']
        if len(quote_positions) % 2 == 1:  # Odd number of quotes means unterminated string
            # Add a closing quote at the end or before the comma/bracket
            comma_pos = line.rfind(',')
            bracket_pos = line.rfind('}')
            colon_pos = line.rfind(':')
            
            # Determine where to add the quote
            if comma_pos > quote_positions[-1]:
                line = line[:comma_pos] + '"' + line[comma_pos:]
            elif bracket_pos > quote_positions[-1]:
                line = line[:bracket_pos] + '"' + line[bracket_pos:]
            else:
                line = line + '"'
        
        repaired_lines.append(line)
    
    # Another pass to fix issues with unterminated strings across multiple lines
    repaired_json = '\n'.join(repaired_lines)
    
    # Balance quotes in the entire string
    if repaired_json.count('"') % 2 == 1:
        repaired_json = repaired_json + '"'
    
    return repaired_json

def repair_unterminated_strings_v2(json_str: str, error_position: int) -> str:
    """
    Advanced repair of unterminated strings in JSON using the error position
    
    Args:
        json_str: JSON string that might contain unterminated strings
        error_position: Position where the error was detected
        
    Returns:
        Repaired JSON string
    """
    # First apply standard repairs
    repaired = repair_unterminated_strings(json_str)
    
    # If standard repairs didn't fix it, try more surgical approach
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError as e:
        if "Unterminated string" not in str(e):
            # If it's not an unterminated string issue anymore, return the repaired version
            return repaired
            
        # Look backwards from error position to find the last quote
        last_quote_pos = json_str.rfind('"', 0, error_position)
        if last_quote_pos != -1:
            # Insert a quote at the error position
            return json_str[:error_position] + '"' + json_str[error_position:]
        
        # If we can't find a quote before the error, general repair
        return repaired

def extract_json_objects_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract individual JSON objects from text and rebuild into a list
    
    Args:
        text: Text possibly containing JSON objects
        
    Returns:
        List of dictionaries
    """
    # Try to find all JSON-like objects
    object_pattern = r'\{\s*"[^"]+"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]+"\s*:\s*(?:"[^"]*"|[^,}]*))* *\}'
    objects = re.findall(object_pattern, text)
    
    result = []
    for obj_str in objects:
        try:
            # Clean and fix the object string
            cleaned_obj = clean_json_string(obj_str)
            repaired_obj = repair_unterminated_strings(cleaned_obj)
            obj = json.loads(repaired_obj)
            result.append(obj)
        except Exception:
            continue
    
    # If we found at least the expected objects, return them
    if len(result) >= 2:
        return result
    return None

def build_json_from_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Build JSON from scratch using regex patterns
    
    Args:
        text: Text to extract patterns from
        
    Returns:
        List of dictionaries built from patterns
    """
    print("Attempting to build JSON from patterns in the text...")
    result = manually_fix_json(text)
    return result

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by removing invalid control characters and fixing common
    formatting issues that cause parsing errors
    
    Args:
        json_str: Raw JSON string that might contain invalid characters
        
    Returns:
        Cleaned JSON string
    """
    # Remove control characters
    json_str = ''.join(ch for ch in json_str if ch.isprintable())
    
    # Fix common issues with YES/NO not being properly quoted
    json_str = re.sub(r':\s*YES\s*,', ': "YES",', json_str)
    json_str = re.sub(r':\s*NO\s*,', ': "NO",', json_str)
    json_str = re.sub(r':\s*YES\s*\}', ': "YES"}', json_str)
    json_str = re.sub(r':\s*NO\s*\}', ': "NO"}', json_str)
    
    # Fix trailing commas in arrays
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing quotes around keys
    json_str = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
    
    # Fix common issue with unescaped quotes inside strings
    # This regex looks for patterns like: "text"more text" and fixes to "text\"more text"
    pattern = r'(?<=[^\\])"(?=[^"]*"[^"]*$)'
    # Apply multiple times to catch nested issues
    for _ in range(3):
        json_str = re.sub(pattern, '\\"', json_str)
    
    return json_str

def manually_fix_json(text: str) -> List[Dict[str, Any]]:
    """
    Attempt to manually fix broken JSON in the model response
    
    Args:
        text: Raw response text from the model
        
    Returns:
        List of dictionaries if successful, None otherwise
    """
    # Common patterns in the response
    question_pattern = r'"Question[12]"\s*:\s*"([^"]+)"'
    explanation_pattern = r'"Explanation[2]?"\s*:\s*"([^"]*)"'
    source_pattern = r'"Source"\s*:\s*"([^"]*)"'
    answer_pattern = r'"Answer"\s*:\s*(?:"([^"]*)"|([A-Z]+))'
    tags_pattern = r'"Tags"\s*:\s*\[(.*?)\]'
    
    # Extract elements
    questions = re.findall(question_pattern, text)
    explanations = re.findall(explanation_pattern, text)
    sources = re.findall(source_pattern, text)
    
    # Extract answers, considering both quoted and unquoted
    answers = []
    for answer_match in re.finditer(answer_pattern, text):
        quoted_answer = answer_match.group(1)
        unquoted_answer = answer_match.group(2)
        answers.append(quoted_answer if quoted_answer else unquoted_answer)
    
    # Extract and clean tags
    tags_list = []
    for tags_match in re.finditer(tags_pattern, text):
        tags_str = tags_match.group(1)
        # Extract individual tags
        tags = [tag.strip().strip('"') for tag in tags_str.split(',')]
        tags_list.append(tags)
    
    # Build JSON structure
    result = []
    if len(questions) >= 1 and len(explanations) >= 1:
        result.append({
            "Question1": questions[0],
            "Explanation": explanations[0],
            "Source": sources[0] if sources else "",
            "Answer": answers[0] if answers else "YES",
            "Tags": tags_list[0] if tags_list else []
        })
    
    if len(questions) >= 2 and len(explanations) >= 2:
        # Check if we have Explanation2 or a second Explanation
        explanation_key = "Explanation2"
        explanation_text = explanations[1] if len(explanations) > 1 else ""
        
        result.append({
            "Question2": questions[1],
            explanation_key: explanation_text,
            "Source": sources[1] if len(sources) > 1 else (sources[0] if sources else ""),
            "Answer": answers[1] if len(answers) > 1 else (answers[0] if answers else "YES"),
            "Tags": tags_list[1] if len(tags_list) > 1 else (tags_list[0] if tags_list else [])
        })
    
    return result if result else None

# API and model configuration data
api_data = [
    {"Thread": "Thread1", "API": "GET IT FROM GOOGLE GEN AI - GOOGLE STUDIO", "Model": "gemini-2.5-flash-preview-04-17"},
     
]

# Rate limiting setup
REQUESTS_PER_MINUTE = 10  # Rate limit: requests per minute PER THREAD
REQUESTS_PER_THREAD = 1500  # Maximum number of requests per thread
rate_limit_lock = threading.Lock()
thread_request_counters = {}  # Track requests per thread
thread_request_timestamps = {}  # Track request timestamps separately for each thread

# Thread-safe console output
print_lock = threading.Lock()
def safe_print(message):
    with print_lock:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


import re
import json
from ftfy import fix_text
from unidecode import unidecode

def clean_json_string(raw_text):
    # Fix Unicode encoding issues
    text = fix_text(raw_text)

    # Convert to ASCII where possible
    text = unidecode(text)

    # Remove non-printable characters (control characters)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Remove odd line breaks
    text = text.replace('\\\n', '').replace('\r', '')

    # Escape double quotes inside JSON strings
    text = re.sub(r'(?<!\\)"', r'\\"', text)

    # Normalize unescaped newlines to \\n (within JSON string values)
    text = re.sub(r'(?<!\\)\n', r'\\n', text)

    # Optionally collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()

def process_question(row, thread_id, api_key, model_name):
    """
    Process a single question with the specified API key and model
    
    Args:
        row: Row from the DataFrame containing question data
        thread_id: Thread identifier
        api_key: API key for this thread
        model_name: Model name for this thread
        
    Returns:
        Tuple of (question_count, success_status)
    """
    # Get question count early to use in messages
    question_count = row['QuestionCount']
    
    # Define output directory and output file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_folder = os.path.join(base_dir, "Questions")
    output_dir = os.path.join(question_folder, str(question_count))
    output_path = os.path.join(output_dir, "Question.json")
    
    # Check if output file already exists - only skip if the JSON file exists, not just the directory
    if os.path.exists(output_path):
        # Validate that the existing JSON is properly formatted
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                json.loads(json_content)  # Try to parse the JSON
                safe_print(f"Thread-{thread_id}: Valid Question.json already exists for question {question_count} - Skipping")
                return question_count, "SKIPPED"
        except json.JSONDecodeError:
            safe_print(f"Thread-{thread_id}: Invalid JSON found in Question.json for question {question_count} - Regenerating")
            # We'll continue and regenerate this question
        except Exception as e:
            safe_print(f"Thread-{thread_id}: Error reading Question.json for question {question_count}: {str(e)} - Regenerating")
            # We'll continue and regenerate this question
        
    # Check if ProcessedText.md exists
    markdown_path = os.path.join(question_folder, str(question_count), "ProcessedText.md")
    if not os.path.exists(markdown_path):
        safe_print(f"Thread-{thread_id}: ProcessedText.md not found for question {question_count} - Skipping")
        return question_count, False
    
    # Directory may exist without Question.json, let's log this information
    if os.path.exists(output_dir) and not os.path.exists(output_path):
        safe_print(f"Thread-{thread_id}: Directory exists for question {question_count}, but Question.json is missing - Processing")
    
    # Now initialize thread counter and apply rate limiting PER THREAD - ONLY when we're about to make an API request
    with rate_limit_lock:
        # Initialize counters and timestamp arrays for this thread if they don't exist
        if thread_id not in thread_request_counters:
            thread_request_counters[thread_id] = 0
            thread_request_timestamps[thread_id] = []
        
        # Check if this thread has reached its limit
        if thread_request_counters[thread_id] >= REQUESTS_PER_THREAD:
            safe_print(f"Thread-{thread_id}: Thread has reached its request limit of {REQUESTS_PER_THREAD} - Exiting")
            return None, "THREAD_LIMIT_REACHED"
        
        # Implement rate limiting PER THREAD
        current_time = time.time()
        thread_timestamps = thread_request_timestamps[thread_id]
        
        # Clean up old timestamps (older than 60 seconds) for this thread
        while thread_timestamps and current_time - thread_timestamps[0] > 60:
            thread_timestamps.pop(0)
            
        # If this thread has reached its own rate limit in the last minute, wait
        if len(thread_timestamps) >= REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - thread_timestamps[0])
            safe_print(f"Thread-{thread_id}: Thread rate limit reached, waiting {wait_time:.2f} seconds before processing question {question_count}")
            time.sleep(wait_time)
            # Recalculate current time after waiting
            current_time = time.time()
            
            # Clean up old timestamps again after waiting
            while thread_timestamps and current_time - thread_timestamps[0] > 60:
                thread_timestamps.pop(0)
        
        # Record this request and increment the counter for this thread
        thread_timestamps.append(current_time)
        thread_request_counters[thread_id] += 1
        safe_print(f"Thread-{thread_id}: Processing question {question_count} (Request #{thread_request_counters[thread_id]}/{REQUESTS_PER_THREAD})")
    
    # Configure the API for this thread
    try:
        # Each thread uses its own genai configuration
        thread_genai = genai
        thread_genai.configure(api_key=api_key)
        thread_model = thread_genai.GenerativeModel(model_name)
    except Exception as e:
        safe_print(f"Thread-{thread_id}: Error configuring API: {str(e)}")
        # Don't save error, just return failure
        return question_count, False
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as file:
            content = file.read()
    
        prompt = f"""
        You are an expert in extracting questions and answers from Biostars. Your task is to extract 
        questions and answers from the Biostars online forum, where users post queries, problems, and images, 
        and others respond in the form of comments. Some comments may contain accurate answers, and those answers have high upvotes.

        ### **1. Extraction Rules**
        - **Ignore** usernames.
        - **Include** references to external tools or links.
        - **Ignore** the number of likes or dislikes.
        - **Choose** the answer based on the comment with the highest number of upvotes and if upvotes are missing then choose the most appropriate answer based on the discussion.
        
        - **Provide** the source link for each answer.
        - **If no valid answer is found,** use "-" in the answer section.
        - **Limit to two questions only.**
        - **Use direct instructions** (e.g., "How to do something?" instead of "How does one do something?").
        - **Ensure answers follow an instructional style** (e.g., "To do this, follow these steps...").
        - **Use a conversational tone** (e.g., "You can do this by...") to make the content more engaging. and do not use "I" or "we" in the answer or the user is asking or the user wants.

        ### **2. Output Format**
        - Format the extracted data as a **valid JSON array**:

        ```json
        [
            {{
                "Question1": "General question title?",
                "Explanation": "General explanation based on the discussion overall discussion", 100+ words based on discussion.
                "Source": "{row['URL']}",
                "Answer":YES/NO, **If there is no answer found in explanation put no otherwise yes. **
                "Tags": ["tag1", "tag2"]
            }},
            {{
                "Question2": "Detailed question based on the issue the user is facing?", **it should be a detailed questions based on the user query.**
                "Explanation2": "Detailed explanation of the problem and its solution.", **500+ words based on discussion. Include code and explanation as well**.
                "Source": "{row['URL']}",
                "Answer":YES/NO, **If there is no answer found in explanation put no otherwise yes. **
                "Tags": ["tag1", "tag2"]
            }}
        ]
        ```

        ### **3. Formatting Guidelines**
        - Use **Markdown triple backticks** (```) for **code snippets**.
        - If applicable, use **tables** for clarity.
        - Maintain **proper indentation and spacing** for readability.

        ### **4. Strict Adherence to Provided Information**
        - **Do not fabricate** any details not explicitly found in the provided text.
        - Stay strictly within the **scope of the given content**.

        ### **5. Citations & References**
        - At the end of each complete answer, **cite the source** as follows:

        ```SOURCE LINK: {row['URL']}```

        ---

        ### **Text from Discussion:**
        {content}

        ---
        """
    
        # Only retry if the response is completely empty or JSON parsing fails
        max_retries = 40
        success = False
        
        for attempt in range(max_retries):
            try:
                safe_print(f"Thread-{thread_id}: Sending request to API for question {question_count} (attempt {attempt+1}/{max_retries})")
                response = thread_model.generate_content(prompt)
                response_text = response.text
                
            

                # Check if response is empty
                if not response_text.strip():
                    if attempt < max_retries - 1:
                        safe_print(f"Thread-{thread_id}: Empty response received, retrying... (attempt {attempt+1}/{max_retries})")
                        time.sleep(2)  # Wait before retry
                        continue
                    else:
                        safe_print(f"Thread-{thread_id}: Empty response received after {max_retries} attempts. Moving to next question.")
                        return question_count, False
                
                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)


                
                # Count occurrences of triple backticks
                backtick_count = response_text.count('```')
                if backtick_count > 2:
                    # Keep only first and last triple backticks
                    first_pos = response_text.find('```')
                    last_pos = response_text.rfind('```')
                    if first_pos != -1 and last_pos != -1 and first_pos != last_pos:
                        cleaned_text = response_text[:first_pos+3] + response_text[first_pos+3:last_pos].replace("```", "'''") + response_text[last_pos:]
                        response_text = cleaned_text
                print(response_text)

                # Try to extract JSON content from triple backticks if present
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_content = json_match.group(1).strip()
                    
                    # Try to parse the extracted JSON
                    try:
                        parsed_json = json.loads(json_content)
                        
                        # If parsing succeeds, write the formatted JSON to file
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(parsed_json, f, indent=2)
                        safe_print(f"Thread-{thread_id}: Extracted and validated JSON content from response for question {question_count}")
                        success = True
                        break
                    except json.JSONDecodeError as json_err:
                        safe_print(f"Thread-{thread_id}: Failed to parse extracted JSON: {str(json_err)}. Retrying...")
                        if attempt < max_retries - 1:
                            continue  # Try again with a new request
                
                # If no JSON in backticks, try to find any JSON-like content
                json_array_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if json_array_match:
                    json_content = json_array_match.group(0)
                    
                    # Try to parse the JSON array
                    try:
                        parsed_json = json.loads(json_content)
                        
                        # If parsing succeeds, write the formatted JSON to file
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(parsed_json, f, indent=2)
                        safe_print(f"Thread-{thread_id}: Extracted and validated JSON array from response for question {question_count}")
                        success = True
                        break
                    except json.JSONDecodeError as json_err:
                        print("response_text", json_content)
                        safe_print(f"Thread-{thread_id}: Failed to parse JSON array: {str(json_err)}. Retrying...")
                        if attempt < max_retries - 1:
                            continue  # Try again with a new request
                
                # If we couldn't extract valid JSON, retry if attempts remain
                safe_print(f"Thread-{thread_id}: Could not extract valid JSON from response for question {question_count}")
                if attempt < max_retries - 1:
                    safe_print(f"Thread-{thread_id}: Retrying to get valid JSON...")
                else:
                    # Last resort: try to manually parse/fix the JSON
                    safe_print(f"Thread-{thread_id}: All API attempts failed to return valid JSON. Trying manual JSON extraction...")
                    
                    # Use the existing manually_fix_json function
                    manual_json = manually_fix_json(response_text)
                    
                    if manual_json:
                        # Save the manually fixed JSON
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(manual_json, f, indent=2)
                        safe_print(f"Thread-{thread_id}: Saved manually fixed JSON for question {question_count}")
                        success = True
                    else:
                        safe_print(f"Thread-{thread_id}: Failed to extract or fix JSON. Moving to next question.")
                        
            except Exception as e:
                safe_print(f"Thread-{thread_id}: API error for question {question_count} (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Wait before retry
                else:
                    safe_print(f"Thread-{thread_id}: All API attempts failed for question {question_count}. Moving to next question.")
        
        return question_count, success
        
    except Exception as e:
        safe_print(f"Thread-{thread_id}: Error processing question {question_count}: {str(e)}")
        # Don't save error message, just return failure
        return question_count, False

def worker_thread(worker_id, task_queue, results, api_key, model_name):
    """
    Worker thread function that processes questions from the queue
    
    Args:
        worker_id: ID of this worker thread
        task_queue: Queue of tasks to process
        results: List to store results
        api_key: API key for this worker
        model_name: Model name for this worker
    """
    safe_print(f"Thread-{worker_id}: Starting with API key: {api_key[:10]}... and model: {model_name}")
    
    while True:
        # Get a task from the queue
        try:
            row = task_queue.get(block=False)
        except queue.Empty:
            safe_print(f"Thread-{worker_id}: No more tasks in queue, exiting")
            break
            
        # Process the task
        question_count, status = process_question(row, worker_id, api_key, model_name)
        
        # Add result to the results list
        if status == "THREAD_LIMIT_REACHED":
            safe_print(f"Thread-{worker_id}: Thread request limit reached ({REQUESTS_PER_THREAD}), exiting")
            break
            
        if question_count is not None:  # Skip if we hit the request limit
            with threading.Lock():
                results.append((question_count, status))
        
        # Mark task as done
        task_queue.task_done()

def generate_questions_multi_threaded(df, num_threads=24, requests_per_thread=1500, requests_per_minute=10):
    """
    Generate questions using multiple threads with different API keys
    
    Args:
        df: DataFrame containing questions data
        num_threads: Number of threads to use (default: 24)
        requests_per_thread: Maximum number of requests per thread (default: 1500)
        requests_per_minute: Maximum requests per minute per thread (default: 10)
    """
    global REQUESTS_PER_THREAD, REQUESTS_PER_MINUTE
    REQUESTS_PER_THREAD = requests_per_thread
    REQUESTS_PER_MINUTE = requests_per_minute
    
    safe_print(f"Starting multi-threaded processing with {num_threads} threads")
    safe_print(f"Request limit set to {REQUESTS_PER_THREAD} per thread ({REQUESTS_PER_THREAD * num_threads} total possible requests)")
    safe_print(f"Rate limit set to {REQUESTS_PER_MINUTE} requests per minute PER THREAD")
    
    # Create a queue for tasks
    task_queue = queue.Queue()
    
    # Add all tasks to the queue
    for _, row in df.iterrows():
        task_queue.put(row)
    
    safe_print(f"Added {task_queue.qsize()} tasks to the queue")
    
    # List to store results
    results = []
    
    # Create and start worker threads
    threads = []
    for i in range(min(num_threads, len(api_data))):
        thread_config = api_data[i]
        thread_id = thread_config["Thread"]
        api_key = thread_config["API"]
        model_name = thread_config["Model"]
        
        t = threading.Thread(
            target=worker_thread,
            args=(thread_id, task_queue, results, api_key, model_name)
        )
        t.start()
        threads.append(t)
        safe_print(f"Started {thread_id} with model {model_name}")
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    # Print summary
    successful = sum(1 for _, status in results if status is True)
    skipped = sum(1 for _, status in results if status == "SKIPPED")
    failed = sum(1 for _, status in results if status is False)
    
    safe_print("\n===== FINAL SUMMARY =====")
    total_requests = sum(thread_request_counters.values())
    safe_print(f"Total requests processed across all threads: {total_requests}")
    for thread_id, count in thread_request_counters.items():
        # Calculate this thread's rate
        if thread_id in thread_request_timestamps and thread_request_timestamps[thread_id]:
            thread_duration = thread_request_timestamps[thread_id][-1] - thread_request_timestamps[thread_id][0]
            if thread_duration > 0:
                rate_per_minute = (count / thread_duration) * 60
                safe_print(f"  - {thread_id}: {count}/{REQUESTS_PER_THREAD} requests (avg rate: {rate_per_minute:.2f} requests/minute)")
            else:
                safe_print(f"  - {thread_id}: {count}/{REQUESTS_PER_THREAD} requests")
        else:
            safe_print(f"  - {thread_id}: {count}/{REQUESTS_PER_THREAD} requests")
    safe_print(f"Successfully processed: {successful}")
    safe_print(f"Skipped (already exist): {skipped}")
    safe_print(f"Failed: {failed}")
    if total_requests > 0:
        success_rate = ((successful + skipped) / total_requests) * 100
        safe_print(f"Success rate: {success_rate:.2f}%")

def verify_json_files(question_folder):
    """
    Verify all Question.json files in the directory for valid JSON format
    
    Args:
        question_folder: Path to the Questions directory
    
    Returns:
        Tuple of (total_files, valid_files, invalid_files, list_of_invalid)
    """
    safe_print("Verifying JSON files...")
    total_files = 0
    valid_files = 0
    invalid_files = 0
    invalid_list = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(question_folder):
        for file in files:
            if file == "Question.json":
                json_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                        json.loads(json_content)  # Try to parse the JSON
                    valid_files += 1
                except json.JSONDecodeError:
                    invalid_files += 1
                    question_id = os.path.basename(root)
                    invalid_list.append(question_id)
                except Exception as e:
                    invalid_files += 1
                    question_id = os.path.basename(root)
                    invalid_list.append(question_id)
    
    safe_print(f"JSON verification complete: {valid_files}/{total_files} valid, {invalid_files} invalid")
    if invalid_files > 0:
        safe_print(f"Invalid JSON files found in question directories: {invalid_list[:10]}...")
        if len(invalid_list) > 10:
            safe_print(f"... and {len(invalid_list) - 10} more")
    
    return total_files, valid_files, invalid_files, invalid_list

def main():
    # Path to the CSV file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "biostars_filtered_questions.csv")
    question_folder = os.path.join(base_dir, "Questions")
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-json":
        # Verify all JSON files in the Questions directory
        verify_json_files(question_folder)
        return
    
    safe_print(f"Using CSV file: {csv_path}")
    safe_print(f"Using question folder: {question_folder}")
    
    # Create the Questions directory if it doesn't exist
    if not os.path.exists(question_folder):
        os.makedirs(question_folder)
        safe_print(f"Created Questions directory: {question_folder}")
    
    # Load the CSV data
    questions = load_questions_data(csv_path)
    print(len(questions))

    if not questions.empty:
        # First verify existing JSON files if they exist
        total_files, valid_files, invalid_files, invalid_list = verify_json_files(question_folder)
        
        # If there are invalid JSON files, prioritize processing them
        if invalid_files > 0:
            safe_print(f"Found {invalid_files} invalid JSON files. Adding them to priority processing.")
            
            # Create a filtered DataFrame with only the invalid questions
            invalid_df = questions[questions['QuestionCount'].astype(str).isin(invalid_list)]
            
            if not invalid_df.empty:
                safe_print(f"Processing {len(invalid_df)} invalid JSON files first")
                generate_questions_multi_threaded(invalid_df, num_threads=5, requests_per_thread=1500, requests_per_minute=1)
        
        # Process all questions (those already valid will be skipped)
        safe_print("Processing all questions")
        generate_questions_multi_threaded(questions, num_threads=5, requests_per_thread=1500, requests_per_minute=1)
    else:
        safe_print("Failed to load questions data from CSV")

if __name__ == "__main__":
    main()
#26871

