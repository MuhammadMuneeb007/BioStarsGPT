import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
import time
import pandas as pd
import os
import multiprocessing
from functools import partial

def extract_time_info(time_text):
    """Extract years from update text like '9.7 years ago'"""
    if not time_text:
        return None
    
    match = re.search(r'(\d+\.?\d*)\s+years?\s+ago', time_text)
    if match:
        return float(match.group(1))
    return None

def scrape_biostars_page_range(start_page, end_page):
    """Scrape a specified range of pages"""
    all_questions = []
    question_count = 0
    
    for page in range(start_page, end_page + 1):
        url = f"https://www.biostars.org/?page={page}"
        print(f"Process {os.getpid()}: Scraping page {page}...")
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access page {page}. Status code: {response.status_code}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        questions = soup.find_all('div', class_='post open item')
        
        for question in questions:
            question_count += 1
            
            # Extract votes
            votes_div = question.select_one('.stats .ui.mini.stat.left.label div')
            votes = votes_div.text if votes_div else "0"
            
            # Extract replies
            replies_div = question.select_one('.stats .ui.label.stat.mini.question div')
            replies = replies_div.text if replies_div else "0"
            
            # Extract views
            views_div = question.select_one('.stats .ui.label.basic.transparent.stat.mini div')
            views = views_div.text.replace('k', '000') if views_div else "0"
            
            # Extract title
            title_elem = question.select_one('.title.mini.header a')
            title = title_elem.text.strip() if title_elem else ""
            # Remove commas from title
            title = title.replace(',', '')
            
            # Extract the URL/link of the question
            question_url = ""
            if title_elem and title_elem.has_attr('href'):
                # Create absolute URL by appending the href to the base URL
                question_url = "https://www.biostars.org" + title_elem['href']
            
            # Extract tags
            tags = []
            tag_elems = question.select('.ptag')
            for tag in tag_elems:
                # Remove commas from tags
                tag_text = tag.text.strip().replace(',', '')
                tags.append(tag_text)
            
            # Extract update time
            update_text = question.select_one('.right.floated.muted').get_text() if question.select_one('.right.floated.muted') else ""
            updated_years = extract_time_info(update_text)
            
            all_questions.append({
                'QuestionCount': question_count,
                'Votes': votes.replace(',', ''),
                'Replies': replies.replace(',', ''),
                'Views': views.replace(',', ''),
                'Title': title,
                'URL': question_url,
                'Tags': '|'.join(tags),
                'Updated_Year': updated_years
            })
        
        # Be nice to the server with a small delay between requests
        if page < end_page:
            time.sleep(1)
    
    return all_questions

def scrape_biostars_questions_multiprocessing(num_pages=5, num_processes=5):
    """Distribute the scraping work across multiple processes"""
    # Calculate how many pages each process should handle
    pages_per_process = num_pages // num_processes
    remainder = num_pages % num_processes
    
    # Create page ranges for each process
    page_ranges = []
    start_page = 1
    
    for i in range(num_processes):
        # Add one extra page to some processes if there's a remainder
        extra = 1 if i < remainder else 0
        end_page = start_page + pages_per_process + extra - 1
        
        page_ranges.append((start_page, end_page))
        start_page = end_page + 1
    
    print(f"Distributing work across {num_processes} processes:")
    for i, (start, end) in enumerate(page_ranges):
        print(f"Process {i+1} will handle pages {start} to {end}")
    
    # Use a multiprocessing pool to distribute the work
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for start, end in page_ranges:
            results.append(pool.apply_async(scrape_biostars_page_range, (start, end)))
        
        # Collect all results
        all_questions = []
        for result in results:
            all_questions.extend(result.get())
    
    # Renumber the questions sequentially
    for i, question in enumerate(all_questions):
        question['QuestionCount'] = i + 1
    
    return all_questions

def save_to_csv(questions, filename="biostars_questions.csv"):
    # Get current working directory and create the filepath
    filepath = os.path.join(os.getcwd(), filename)
    
    # Convert to pandas DataFrame and save to CSV
    df = pd.DataFrame(questions)
    df.to_csv(filepath, index=False)
    
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # This is required for Windows to avoid issues with multiprocessing
    multiprocessing.freeze_support()
    
    print("Scraping Biostars questions using multiprocessing...")
    total_pages = 2418
    processes = 5
    
    questions = scrape_biostars_questions_multiprocessing(total_pages, processes)
    print(f"Found {len(questions)} questions")
    save_to_csv(questions)
