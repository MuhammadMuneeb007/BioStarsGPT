import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import urllib.parse
from urllib.request import urlretrieve
import html2markdown
import easyocr
import numpy as np
import cv2
from PIL import Image
import logging
import markdown2
from bs4 import BeautifulSoup
import re
import html
from markdownify import markdownify as md
import glob  # Add the missing glob import
import concurrent.futures  # For multi-threading
import tqdm  # For progress bar
from threading import Lock  # For thread-safe operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a lock for thread-safe printing
print_lock = Lock()

# Create a thread-safe print function
def safe_print(message):
    with print_lock:
        print(message)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_image(image_url, save_path):
    """Download image from URL and save it to specified path"""
    try:
        # Use requests instead of urlretrieve for better error handling
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(save_path, 'wb') as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                img_file.write(chunk)
                
        print(f"Downloaded image: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")
        return False

def extract_images_from_content(soup, question_folder):
    """Extract images from content and download them"""
    image_count = 0
    image_mapping = {}  # Track original URLs to local filenames
    
    # Look for images in all possible locations
    images = soup.select('img')
    
    for i, img in enumerate(images, 1):
        if 'src' in img.attrs:
            img_url = img['src']
            if not img_url.startswith(('http://', 'https://')):
                img_url = 'https://www.biostars.org' + img_url
            
            # Extract original filename from URL
            parsed_url = urllib.parse.urlparse(img_url)
            original_filename = os.path.basename(parsed_url.path)
            
            # Get file extension or determine from content-type if missing
            file_ext = os.path.splitext(original_filename)[1]
            
            # If no valid extension, try to get it from content-type or default to .png
            if not file_ext or file_ext == '.':
                try:
                    # Make a HEAD request to get content-type
                    head_response = requests.head(img_url, timeout=3)
                    content_type = head_response.headers.get('Content-Type', '')
                    
                    if 'image/jpeg' in content_type or 'image/jpg' in content_type:
                        file_ext = '.jpg'
                    elif 'image/png' in content_type:
                        file_ext = '.png'
                    elif 'image/gif' in content_type:
                        file_ext = '.gif'
                    elif 'image/svg+xml' in content_type:
                        file_ext = '.svg'
                    elif 'image/webp' in content_type:
                        file_ext = '.webp'
                    else:
                        file_ext = '.png'  # Default to .png
                except Exception:
                    file_ext = '.png'  # Default to .png if HEAD request fails
                
                # If no filename or it's just an extension or invalid
                if not original_filename or original_filename == file_ext or "/" in original_filename:
                    original_filename = f"Image{i}{file_ext}"
                elif not os.path.splitext(original_filename)[1]:
                    # If filename exists but has no extension, add the detected extension
                    original_filename = f"{original_filename}{file_ext}"
            
            # Handle potential filename conflicts
            base_name, ext = os.path.splitext(original_filename)
            image_path = os.path.join(question_folder, original_filename)
            counter = 1
            
            # If file already exists, add a suffix
            while os.path.exists(image_path):
                original_filename = f"{base_name}_{counter}{ext}"
                image_path = os.path.join(question_folder, original_filename)
                counter += 1
            
            if download_image(img_url, image_path):
                # Store the mapping from original URL to local filename
                image_mapping[img_url] = original_filename
                image_count += 1
    
    return image_mapping

def sanitize_filename(filename):
    """Sanitize filename by removing problematic characters including backslashes"""
    # Replace backslashes with underscores
    filename = filename.replace('\\', '_')
    # Replace other problematic characters
    filename = re.sub(r'[/*?:"<>|]', '_', filename)
    return filename

def replace_image_urls_in_markdown(markdown_text, image_mapping):
    """Replace image URLs in markdown text with local filenames"""
    for img_url, local_filename in image_mapping.items():
        # Make sure the local filename doesn't have backslashes
        clean_filename = local_filename.replace('\\', '_')
        
        # Handle both markdown and HTML style image references
        # Markdown style: ![alt](url)
        markdown_text = re.sub(r'!\[([^\]]*)\]\(' + re.escape(img_url) + r'\)', 
                             r'![\1](' + clean_filename + r')', markdown_text)
        
        # HTML style: <img src="url" ... >
        markdown_text = re.sub(r'<img[^>]*src=[\'"]' + re.escape(img_url) + r'[\'"][^>]*>', 
                             r'<img src="' + clean_filename + r'" />', markdown_text)
        
        # Handle empty image tags which might be leftovers from html2markdown conversion
        markdown_text = re.sub(r'!\[\]\(\)', r'![Image](' + clean_filename + ')', markdown_text)
        markdown_text = re.sub(r'<img>', r'![Image](' + clean_filename + ')', markdown_text)
    
    # Fix any remaining empty image tags - not mapped to specific images
    markdown_text = re.sub(r'!\[\]\(\)', '', markdown_text)
    markdown_text = re.sub(r'<img>', '', markdown_text)
    
    return markdown_text

def clean_markdown(markdown_text):
    """Clean up the markdown text by removing unnecessary HTML tags and artifacts"""
    # Remove div and span tags that might remain
    markdown_text = re.sub(r'</?div[^>]*>', '', markdown_text)
    markdown_text = re.sub(r'</?span[^>]*>', '', markdown_text)
    
    # Remove class attributes and other HTML artifacts
    markdown_text = re.sub(r'<([a-z]+)[^>]*>', r'<\1>', markdown_text)
    
    # Clean up user box remnants
    markdown_text = re.sub(r'<a>\s*<img>\s*</a>', '', markdown_text)
    
    # Fix double newlines and spacing
    markdown_text = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_text)
    
    # Clean bullet points
    markdown_text = re.sub(r'•', '*', markdown_text)
    
    # Remove any remaining HTML comment tags
    markdown_text = re.sub(r'<!--.*?-->', '', markdown_text)
    
    # Remove empty tags
    markdown_text = re.sub(r'<([a-z]+)>\s*</\1>', '', markdown_text)
    
    # Clean any numeric bullets that might be part of formatting
    markdown_text = re.sub(r'\n\s*\d+\s*\n', '\n\n', markdown_text)
    
    # Clean up any remaining UI icon references
    markdown_text = re.sub(r'<i[^>]*></i>', '', markdown_text)
    
    return markdown_text.strip()

def extract_user_info(element):
    """Extract comprehensive user info from a post or comment with improved error handling"""
    user_info = {'name': "Anonymous", 'date': "Unknown date", 'votes': "0"}
    
    try:
        if element is None:
            print("Warning: Received None element in extract_user_info")
            return user_info
            
        # Extract username
        user_link = element.select_one('.user_box a, .user-info a')
        if user_link and hasattr(user_link, 'text') and user_link.text:
            user_info['name'] = user_link.text.strip()
        
        # Extract date
        date_div = element.select_one('.user_box .muted div:first-child, .user-info .muted')
        if date_div and hasattr(date_div, 'text') and date_div.text:
            user_info['date'] = date_div.text.strip()
        
        # Extract votes/score with better error handling
        try:
            # First look directly in the voting div which is more reliable for answers
            score_div = element.select_one('.voting .score')
            if score_div and hasattr(score_div, 'text') and score_div.text:
                # Try to extract the number directly
                user_info['votes'] = score_div.text.strip()
            else:
                # Look for comment mini-button structure
                mini_button_parent = element.select_one('.voting, .body')
                if mini_button_parent:
                    # Find the score div that appears after the thumbs up mini button
                    thumb_button = mini_button_parent.select_one('button[data-value="upvote"]')
                    if thumb_button and thumb_button.find_next_sibling():
                        score_div = thumb_button.find_next_sibling()
                        if score_div and hasattr(score_div, 'name') and score_div.name == 'div' and 'score' in score_div.get('class', []):
                            if hasattr(score_div, 'text') and score_div.text:
                                user_info['votes'] = score_div.text.strip()
                
                # If we haven't found votes yet, try other common selectors
                if user_info['votes'] == "0":
                    score_div = element.select_one('.vote-count, .score, .vote-number')
                    if score_div and hasattr(score_div, 'text') and score_div.text:
                        user_info['votes'] = score_div.text.strip()
                    else:
                        # Try alternative selectors for comment votes
                        vote_elem = element.select_one('.comment-score, .vote-number, .vote, .score-number')
                        if vote_elem and hasattr(vote_elem, 'text') and vote_elem.text:
                            user_info['votes'] = vote_elem.text.strip()
        except Exception as votes_error:
            print(f"Error extracting votes: {str(votes_error)}")
                
        # Clean up the votes string - remove any non-numeric characters except minus sign
        user_info['votes'] = re.sub(r'[^\d-]', '', user_info.get('votes', '0')) or "0"
        
    except Exception as e:
        print(f"Error in extract_user_info: {str(e)}")
        # Default values already set
        
    return user_info

def fix_markdown_image_references(markdown_text, question_folder):
    """Fix markdown image references that aren't properly linked"""
    # Find all ![Image](filename) patterns where filename doesn't have a proper URL/path
    image_refs = re.findall(r'!\[(.*?)\]\((.*?)\)', markdown_text)
    
    for alt_text, src in image_refs:
        # If the src doesn't have http:// or https:// and doesn't have a file extension
        # it might be a broken reference
        if not src.startswith(('http://', 'https://')) and not re.search(r'\.\w+$', src):
            # Look for actual image files in the folder that might match
            possible_images = [f for f in os.listdir(question_folder) 
                              if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'))]
            
            # Try to find a match by checking if the filename contains the src as a substring
            matches = [img for img in possible_images if src.lower() in img.lower()]
            
            if matches:
                # Use the first match
                new_src = matches[0]
                markdown_text = markdown_text.replace(f'![{alt_text}]({src})', f'![{alt_text}]({new_src})')
                print(f"Fixed image reference: {src} -> {new_src}")
    
    return markdown_text

def process_markdown_for_image_references(markdown_text, image_mapping, question_folder):
    """Process markdown to ensure all image references are correct"""
    # First replace images with known mappings
    markdown_text = replace_image_urls_in_markdown(markdown_text, image_mapping)
    
    # Special case: handle ![Image](filename) where filename is just the name without extension
    # This pattern is common in the data you showed
    pattern = r'!\[Image\]\((.*?)\)'
    for match in re.finditer(pattern, markdown_text):
        filename = match.group(1)
        # Check if this is a simple name without path or extension
        if not re.search(r'[/\\.]', filename):
            # Look for any image file in the folder that starts with this name
            for img_file in os.listdir(question_folder):
                if img_file.lower().startswith(filename.lower()) and os.path.isfile(os.path.join(question_folder, img_file)):
                    # Replace with the actual filename
                    new_ref = f'![Image]({img_file})'
                    markdown_text = markdown_text.replace(match.group(0), new_ref)
                    print(f"Replaced image reference: {match.group(0)} -> {new_ref}")
                    break
    
    # Fix any remaining standard image markdown that might have been missed
    markdown_text = re.sub(r'!\[Image\]\(([^)]+)\)', 
                          lambda m: f'![Image]({os.path.basename(m.group(1))})' 
                          if m.group(1).startswith(('http://', 'https://')) else m.group(0), 
                          markdown_text)
    
    # Attempt to fix any remaining images with plain names
    markdown_text = fix_markdown_image_references(markdown_text, question_folder)
    
    return markdown_text

def extract_comments(post_div, question_folder):
    """Extract comments attached to a post"""
    comment_markdown = ""
    comments = post_div.select('.comment-list .comment')
    
    if comments:
        comment_markdown += "\n#### Comments:\n\n"
        for i, comment in enumerate(comments, 1):
            # Extract user info with improved vote extraction
            user_info = extract_user_info(comment)
            
            # Extract comment content
            content_div = comment.select_one('.magnify')
            if content_div:
                # Extract images directly from HTML
                image_mapping = directly_extract_images_from_html(str(content_div), question_folder)
                
                # Convert to markdown
                content_html = str(content_div)
                content_md = html2markdown.convert(content_html)
                content_md = clean_markdown(content_md)
                
                # Replace image URLs in the markdown with local filenames
                for img_url, local_filename in image_mapping.items():
                    content_md = content_md.replace(f'![]({img_url})', f'![Image]({local_filename})')
                    content_md = content_md.replace(f'![Image]({img_url})', f'![Image]({local_filename})')
                    
                    # Check for the specific pattern in your example
                    base_url = '/media/images/'
                    if img_url.startswith('https://www.biostars.org' + base_url):
                        img_id = img_url.replace('https://www.biostars.org' + base_url, '')
                        content_md = content_md.replace(f'![Image]({base_url}{img_id})', f'![Image]({local_filename})')
                
                # One more pass to find any missed images
                image_pattern = r'!\[(.*?)\]\(/media/images/([^)]+)\)'
                matches = re.findall(image_pattern, content_md)
                
                for alt_text, img_id in matches:
                    full_url = f"https://www.biostars.org/media/images/{img_id}"
                    if full_url in image_mapping:
                        local_filename = image_mapping[full_url]
                        content_md = content_md.replace(f'![{alt_text}](/media/images/{img_id})', f'![{alt_text}]({local_filename})')
                
                # Include votes in the comment header
                votes_text = f" | Votes: {user_info['votes']}" if user_info['votes'] != "0" else ""
                comment_markdown += f"**Comment {i}** by {user_info['name']} ({user_info['date']}{votes_text}):\n"
                comment_markdown += f"{content_md}\n\n"
    
    return comment_markdown

def extract_answer_content(answer_div, question_folder):
    """Extract answer content including comments and all metadata"""
    # Extract user info
    user_info = extract_user_info(answer_div)
    
    # Extract the content
    content_div = answer_div.select_one('.content .wrap')
    
    answer_markdown = ""
    if content_div:
        # Extract images directly from HTML
        image_mapping = directly_extract_images_from_html(str(content_div), question_folder)
        
        # Convert to markdown
        content_html = str(content_div)
        answer_markdown = html2markdown.convert(content_html)
        answer_markdown = clean_markdown(answer_markdown)
        
        # Replace image URLs in the markdown with local filenames
        for img_url, local_filename in image_mapping.items():
            answer_markdown = answer_markdown.replace(f'![]({img_url})', f'![Image]({local_filename})')
            answer_markdown = answer_markdown.replace(f'![Image]({img_url})', f'![Image]({local_filename})')
            
            # Check for the specific pattern in your example
            base_url = '/media/images/'
            if img_url.startswith('https://www.biostars.org' + base_url):
                img_id = img_url.replace('https://www.biostars.org' + base_url, '')
                answer_markdown = answer_markdown.replace(f'![Image]({base_url}{img_id})', f'![Image]({local_filename})')
        
        # One more pass to find any missed images
        image_pattern = r'!\[(.*?)\]\(/media/images/([^)]+)\)'
        matches = re.findall(image_pattern, answer_markdown)
        
        for alt_text, img_id in matches:
            full_url = f"https://www.biostars.org/media/images/{img_id}"
            if full_url in image_mapping:
                local_filename = image_mapping[full_url]
                answer_markdown = answer_markdown.replace(f'![{alt_text}](/media/images/{img_id})', f'![{alt_text}]({local_filename})')
        
        # Add user info and votes as metadata
        answer_metadata = f"**Posted by:** {user_info['name']} | **Date:** {user_info['date']} | **Votes:** {user_info['votes']}\n\n"
        answer_markdown = answer_metadata + answer_markdown
        
        # Extract comments on this answer
        comments_markdown = extract_comments(answer_div, question_folder)
        if comments_markdown:
            answer_markdown += comments_markdown
    else:
        answer_markdown = "*No content found in this answer*"
    
    return answer_markdown

def extract_images_and_update_html(soup, question_folder):
    """Extract images and update their references in the HTML before markdown conversion"""
    image_mapping = {}
    
    # Look for images in all possible locations
    images = soup.select('img')
    
    for i, img in enumerate(images, 1):
        if 'src' in img.attrs:
            img_url = img['src']
            
            # Fix URL construction by ensuring proper formatting
            if not img_url.startswith(('http://', 'https://')):
                # Make sure we don't have double slashes that aren't part of the protocol
                if img_url.startswith('/'):
                    img_url = 'https://www.biostars.org' + img_url
                else:
                    img_url = 'https://www.biostars.org/' + img_url
            
            try:
                # Check if the URL is valid
                test_response = requests.head(img_url, timeout=3)
                if test_response.status_code != 200:
                    print(f"Image URL returns error status code {test_response.status_code}: {img_url}")
                    continue
            except Exception as e:
                print(f"Invalid image URL: {img_url}, Error: {e}")
                
                # Try an alternative URL format
                if 'biostars.org' in img_url:
                    # Extract the filename part
                    filename = os.path.basename(img_url)
                    alternative_url = f"https://www.biostars.org/static/images/{filename}"
                    
                    try:
                        test_response = requests.head(alternative_url, timeout=3)
                        if test_response.status_code == 200:
                            print(f"Using alternative URL: {alternative_url}")
                            img_url = alternative_url
                        else:
                            continue
                    except:
                        print(f"Alternative URL also failed: {alternative_url}")
                        continue
                else:
                    continue
            
            # Extract original filename from URL
            parsed_url = urllib.parse.urlparse(img_url)
            original_filename = os.path.basename(parsed_url.path)
            
            # Get file extension or determine from content-type if missing
            file_ext = os.path.splitext(original_filename)[1]
            
            # If no valid extension, try to get it from content-type or default to .png
            if not file_ext or file_ext == '.':
                try:
                    # Make a HEAD request to get content-type
                    head_response = requests.head(img_url, timeout=3)
                    content_type = head_response.headers.get('Content-Type', '')
                    
                    if 'image/jpeg' in content_type or 'image/jpg' in content_type:
                        file_ext = '.jpg'
                    elif 'image/png' in content_type:
                        file_ext = '.png'
                    elif 'image/gif' in content_type:
                        file_ext = '.gif'
                    elif 'image/svg+xml' in content_type:
                        file_ext = '.svg'
                    elif 'image/webp' in content_type:
                        file_ext = '.webp'
                    else:
                        file_ext = '.png'  # Default to .png
                except Exception:
                    file_ext = '.png'  # Default to .png if HEAD request fails
            
            # If no filename or it's just an extension or invalid
            if not original_filename or original_filename == file_ext or "/" in original_filename:
                original_filename = f"Image{i}{file_ext}"
            elif not os.path.splitext(original_filename)[1]:
                # If filename exists but has no extension, add the detected extension
                original_filename = f"{original_filename}{file_ext}"
            
            # Sanitize filename by replacing problematic characters
            original_filename = re.sub(r'[\\/*?:"<>|]', '_', original_filename)
            
            # Handle potential filename conflicts
            base_name, ext = os.path.splitext(original_filename)
            image_path = os.path.join(question_folder, original_filename)
            counter = 1
            
            # If file already exists, add a suffix
            while os.path.exists(image_path):
                original_filename = f"{base_name}_{counter}{ext}"
                image_path = os.path.join(question_folder, original_filename)
                counter += 1
            
            try:
                if download_image(img_url, image_path):
                    # Update the img src in the HTML to the local filename
                    img['src'] = original_filename
                    # Add alt text if missing
                    if not img.get('alt'):
                        img['alt'] = f"Image {i}"
                    # Store the mapping from original URL to local filename
                    image_mapping[img_url] = original_filename
            except Exception as e:
                print(f"Failed to download image {img_url}: {e}")
                continue
    
    return image_mapping

def directly_extract_images_from_html(html_content, question_folder):
    """Extract image URLs directly from HTML content before conversion to markdown"""
    image_mapping = {}
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all img tags
    images = soup.find_all('img')
    
    for i, img in enumerate(images, 1):
        if 'src' in img.attrs:
            img_url = img['src']
            
            # Fix URL construction by ensuring proper formatting
            if not img_url.startswith(('http://', 'https://')):
                # Make sure we don't have double slashes that aren't part of the protocol
                if img_url.startswith('/'):
                    img_url = 'https://www.biostars.org' + img_url
                else:
                    img_url = 'https://www.biostars.org/' + img_url
            
            # Try to download the image
            try:
                # Extract original filename from URL or use alt text if available
                parsed_url = urllib.parse.urlparse(img_url)
                original_filename = os.path.basename(parsed_url.path)
                
                # If alt text exists, use it to create a more descriptive filename
                alt_text = img.get('alt', '')
                if alt_text and len(alt_text) > 3 and len(alt_text) < 50:
                    # Create a filename from alt text - sanitize it first
                    alt_text = re.sub(r'[\\/*?:"<>|]', '_', alt_text)
                    alt_text = alt_text.replace(' ', '_').lower()
                    
                    # Get extension from original or default
                    file_ext = os.path.splitext(original_filename)[1]
                    if not file_ext or file_ext == '.':
                        file_ext = '.png'  # Default extension
                        
                    original_filename = f"{alt_text}{file_ext}"
                
                # If original filename is empty or too short, create one
                if not original_filename or len(original_filename) < 3:
                    original_filename = f"image_{i}.png"
                
                # Sanitize filename
                original_filename = re.sub(r'[\\/*?:"<>|]', '_', original_filename)
                
                # Get file extension or determine from content-type if missing
                file_ext = os.path.splitext(original_filename)[1]
                if not file_ext or file_ext == '.':
                    try:
                        # Make a HEAD request to get content-type
                        head_response = requests.head(img_url, timeout=3)
                        content_type = head_response.headers.get('Content-Type', '')
                        
                        if 'image/jpeg' in content_type or 'image/jpg' in content_type:
                            file_ext = '.jpg'
                        elif 'image/png' in content_type:
                            file_ext = '.png'
                        elif 'image/gif' in content_type:
                            file_ext = '.gif'
                        elif 'image/svg+xml' in content_type:
                            file_ext = '.svg'
                        elif 'image/webp' in content_type:
                            file_ext = '.webp'
                        else:
                            file_ext = '.png'  # Default to .png
                            
                        # Update the filename with the correct extension
                        base_name = os.path.splitext(original_filename)[0]
                        original_filename = f"{base_name}{file_ext}"
                    except Exception:
                        # If we can't determine the type, just append .png
                        if not original_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
                            original_filename += '.png'
                
                # Handle potential filename conflicts
                base_name, ext = os.path.splitext(original_filename)
                image_path = os.path.join(question_folder, original_filename)
                counter = 1
                
                # If file already exists, add a suffix
                while os.path.exists(image_path):
                    original_filename = f"{base_name}_{counter}{ext}"
                    image_path = os.path.join(question_folder, original_filename)
                    counter += 1
                
                # Download the image
                if download_image(img_url, image_path):
                    # Store the mapping from original URL to local filename
                    image_mapping[img_url] = original_filename
                    print(f"Downloaded and mapped: {img_url} -> {original_filename}")
                
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
    
    return image_mapping

def improved_html_to_markdown(html_content, image_mapping=None):
    """
    Convert HTML to clean, properly formatted Markdown with better handling
    of code blocks, lists, and other elements
    """
    if image_mapping is None:
        image_mapping = {}
        
    # Pre-process HTML to fix common issues
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Fix code blocks - ensure proper formatting
    for pre in soup.find_all('pre'):
        # Add a newline before and after code blocks
        if pre.parent and pre.parent.name != 'div':
            pre.insert_before('\n')
            pre.insert_after('\n')
        
        # Make sure code blocks have proper code tags
        if not pre.find('code'):
            code_tag = soup.new_tag('code')
            code_tag.string = pre.string
            pre.string = ''
            pre.append(code_tag)
            
    # Fix lists - ensure proper formatting
    for ul in soup.find_all(['ul', 'ol']):
        # Add a newline before and after lists
        ul.insert_before('\n')
        ul.insert_after('\n')
    
    # Process images with proper alt text and references
    for img in soup.find_all('img'):
        src = img.get('src', '')
        alt = img.get('alt', 'Image')
        
        # Create full URL for relative paths
        if src and not src.startswith(('http://', 'https://')):
            if src.startswith('/'):
                full_src = f"https://www.biostars.org{src}"
            else:
                full_src = f"https://www.biostars.org/{src}"
        else:
            full_src = src
        
        # Check if we have a local filename for this image
        local_filename = image_mapping.get(full_src, '')
        if local_filename:
            # Create a new markdown image reference
            img_markdown = f"![{alt}]({local_filename})"
            new_tag = soup.new_tag('div')
            new_tag.string = img_markdown
            img.replace_with(new_tag)
    
    # Use markdownify for better conversion
    markdown_text = md(str(soup), heading_style="ATX")
    
    # Post-process markdown
    markdown_text = post_process_markdown(markdown_text, image_mapping)
    
    return markdown_text

def post_process_markdown(markdown_text, image_mapping):
    """Clean up and improve markdown after conversion"""
    
    # Replace HTML entities
    markdown_text = html.unescape(markdown_text)
    
    # Fix code blocks - ensure they have proper spacing
    markdown_text = re.sub(r'```\s*\n*', '```\n', markdown_text)
    markdown_text = re.sub(r'\n*\s*```', '\n```', markdown_text)
    
    # Fix heading spacing
    markdown_text = re.sub(r'(#{1,6})\s*([^\n]+)', r'\1 \2', markdown_text)
    
    # Fix list item spacing
    markdown_text = re.sub(r'(\n[*\-+])\s*([^\n]+)', r'\1 \2', markdown_text)
    
    # Fix numbered list spacing
    markdown_text = re.sub(r'(\n\d+\.)\s*([^\n]+)', r'\1 \2', markdown_text)
    
    # Fix multiple newlines (more than 2)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    # Fix any remaining image references
    for img_url, local_filename in image_mapping.items():
        # Ensure the local filename has no backslashes
        clean_filename = local_filename.replace('\\', '_')
        
        # Handle various image formats
        markdown_text = re.sub(r'!\[(.*?)\]\(' + re.escape(img_url) + r'\)', 
                            r'![\1](' + clean_filename + r')', markdown_text)
        
        # Handle relative URLs
        if img_url.startswith('https://www.biostars.org/'):
            relative_url = img_url.replace('https://www.biostars.org', '')
            markdown_text = re.sub(r'!\[(.*?)\]\(' + re.escape(relative_url) + r'\)', 
                                r'![\1](' + clean_filename + r')', markdown_text)
    
    # Clean up any remaining HTML tags
    markdown_text = re.sub(r'<(?!code|pre|\/code|\/pre)[^>]*>', '', markdown_text)
    
    # Fix blockquotes
    markdown_text = re.sub(r'(\n> [^\n]+)(\n)(?!\n> )', r'\1\2\n', markdown_text)
    
    # Clean up backslashes in image references - this is a new step
    markdown_text = clean_image_references(markdown_text)
    
    return markdown_text.strip()

def clean_image_references(markdown_text):
    """Remove backslashes from image references in markdown text"""
    # This regex identifies image references with detailed pattern matching to catch all cases
    # It looks for ![alt](filename) with any possible backslash variations
    markdown_text = markdown_text.replace("\\", "")
    markdown_text = markdown_text.replace("/", "")
    
    
    image_pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_image_ref(match):
        alt_text = match.group(1)
        filename = match.group(2)
        
        # Remove ALL backslashes from filename
        clean_filename = filename.replace('\\', '')
        
        return f"![{alt_text}]({clean_filename})"
    
    # Find and clean all image references
    cleaned_text = re.sub(image_pattern, replace_image_ref, markdown_text)
    return cleaned_text

def fix_markdown_files(directory_path):
    """Fix existing markdown files by removing backslashes in image references"""
    print(f"Fixing markdown files in {directory_path}...")
    
    # Count how many files were fixed
    files_fixed = 0
    references_fixed = 0
    
    # Process all Text.md files in the Questions directory structure
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith("Text.md") or file.endswith("ProcessedText.md"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count escaped image references
                escaped_references = re.findall(r'!\[.*?\]\(.*?\\.*?\)', content)
                if escaped_references:
                    # Clean the content
                    cleaned_content = clean_image_references(content)
                    
                    # Write the cleaned content back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    files_fixed += 1
                    references_fixed += len(escaped_references)
                    print(f"  - Fixed {len(escaped_references)} image references")
    
    print(f"Fixed {references_fixed} image references in {files_fixed} files.")
    return files_fixed, references_fixed

# Replace the custom_html_to_markdown function with our improved version
def custom_html_to_markdown(html_content, image_mapping=None):
    return improved_html_to_markdown(html_content, image_mapping)

def extract_and_download_images(html_content, question_folder):
    """Extract images from HTML content, download them, and create a mapping"""
    soup = BeautifulSoup(html_content, 'html.parser')
    image_mapping = {}
    
    for i, img in enumerate(soup.find_all('img'), 1):
        src = img.get('src', '')
        if not src:
            continue
            
        # Handle relative URLs
        if not src.startswith(('http://', 'https://')):
            if src.startswith('/'):
                src = f"https://www.biostars.org{src}"
            else:
                src = f"https://www.biostars.org/{src}"
        
        try:
            # Extract original filename and alt text
            alt_text = img.get('alt', '').strip()
            parsed_url = urllib.parse.urlparse(src)
            original_filename = os.path.basename(parsed_url.path)
            
            # If the original filename is not useful (empty, too short, no extension)
            if not original_filename or len(original_filename) < 3:
                # Try to use the image ID from media/images URLs
                if '/media/images/' in src:
                    img_id = src.split('/media/images/')[-1]
                    original_filename = img_id
                else:
                    # Use alt text if available for a more descriptive filename
                    if alt_text and len(alt_text) > 2:
                        base_name = re.sub(r'[^\w\s-]', '', alt_text.lower())
                        base_name = re.sub(r'[-\s]+', '-', base_name).strip('-')
                        if base_name:
                            original_filename = base_name
                        else:
                            original_filename = f"image_{i}"
                    else:
                        original_filename = f"image_{i}"
            
            # Get or determine file extension
            file_ext = os.path.splitext(original_filename)[1]
            if not file_ext or file_ext == '.':
                # Try to determine from content-type
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    head_response = requests.head(src, timeout=5, headers=headers)
                    content_type = head_response.headers.get('Content-Type', '')
                    
                    if 'image/jpeg' in content_type or 'image/jpg' in content_type:
                        file_ext = '.jpg'
                    elif 'image/png' in content_type:
                        file_ext = '.png'
                    elif 'image/gif' in content_type:
                        file_ext = '.gif'
                    elif 'image/svg+xml' in content_type:
                        file_ext = '.svg'
                    elif 'image/webp' in content_type:
                        file_ext = '.webp'
                    else:
                        file_ext = '.png'  # Default
                except Exception:
                    file_ext = '.png'  # Default if head request fails
            
            # Ensure filename has extension and sanitize it
            base_name = os.path.splitext(original_filename)[0]
            safe_base_name = sanitize_filename(base_name)
            safe_filename = f"{safe_base_name}{file_ext}"
            
            # Handle filename conflicts
            image_path = os.path.join(question_folder, safe_filename)
            counter = 1
            while os.path.exists(image_path):
                safe_filename = f"{safe_base_name}_{counter}{file_ext}"
                image_path = os.path.join(question_folder, safe_filename)
                counter += 1
            
            # Download the image
            if download_image(src, image_path):
                image_mapping[src] = safe_filename
                print(f"Downloaded: {src} -> {safe_filename}")
                
        except Exception as e:
            print(f"Error processing image {src}: {e}")
    
    return image_mapping

def extract_text_from_image(image_path):
    """Extract text from an image using EasyOCR with improved error handling"""
    try:
        # Initialize the OCR reader only once if not already initialized
        if not hasattr(extract_text_from_image, 'reader'):
            print(f"Initializing EasyOCR reader for image: {image_path}")
            try:
                # Check for GPU availability
                import torch
                gpu_available = torch.cuda.is_available()
                extract_text_from_image.reader = easyocr.Reader(['en'], gpu=gpu_available)
                device = "GPU" if gpu_available else "CPU"
                print(f"EasyOCR initialized using {device}")
            except Exception as e:
                print(f"Error initializing EasyOCR with GPU, falling back to CPU: {e}")
                extract_text_from_image.reader = easyocr.Reader(['en'], gpu=False)
        
        # Check if image exists and is valid
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return "[Image: File not found]"
            
        # Try different methods to read the image
        try:
            # First try with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # If OpenCV fails, try with PIL
                print(f"OpenCV failed to read image, trying PIL: {image_path}")
                image = np.array(Image.open(image_path).convert('RGB'))
                if image.size == 0:
                    return "[Image: Could not be read]"
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return f"[Image: Reading error - {str(e)}]"
            
        # Use EasyOCR to extract text
        results = extract_text_from_image.reader.readtext(image)
        
        if not results:
            print(f"No text detected in image: {image_path}")
            return "[Image: No text detected]"
            
        # Extract text with position information
        extracted_text = ""
        for (bbox, text, prob) in results:
            if prob > 0.2:  # Only include text with reasonable confidence
                extracted_text += text + " "
                
        if not extracted_text.strip():
            return "[Image: No reliable text detected]"
            
        print(f"Successfully extracted {len(extracted_text)} characters from image")
        return f"[OCR Text: {extracted_text.strip()}]"
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        return f"[Image: OCR error - {str(e)}]"
import re
import os
 
 
def create_processed_text_md(markdown_content, question_folder):
    """Create a processed version of the markdown with OCR text replacing images"""
    print(f"Creating processed text with OCR for folder: {question_folder}")
    processed_content = markdown_content
    
    # Fix backslash escaping in image filenames before processing
    # This regex matches markdown image syntax and captures the alt text and filename
    image_pattern = r'!\[(.*?)\]\(([^)]+)\)'
    
    # Process each image reference
    for match in re.finditer(image_pattern, markdown_content):
        alt_text = match.group(1)
        original_filename = match.group(2)
        
        # Clean the filename by removing escaping backslashes
        clean_filename = original_filename.replace('\\', '')
        
        # If the filename had backslashes, update the reference in the markdown
        if '\\' in original_filename:
            old_ref = f"![{alt_text}]({original_filename})"
            new_ref = f"![{alt_text}]({clean_filename})"
            processed_content = processed_content.replace(old_ref, new_ref)
            print(f"Fixed image reference: {original_filename} -> {clean_filename}")
    
    # Now find all images for OCR processing
    image_refs = re.findall(image_pattern, processed_content)
    
    processed_images = 0
    for alt_text, image_filename in image_refs:
        # Make sure there are no backslashes in the filename
        clean_filename = image_filename.replace('\\', '')
        
        print(f"Processing image reference: {clean_filename}")
        
        # Try different path combinations to find the actual image file
        possible_paths = [
            os.path.join(question_folder, clean_filename),  # Clean filename
            os.path.join(question_folder, os.path.basename(clean_filename))  # Just the basename
        ]
        
        # Also check for any image in the folder with the same basename without extension
        base_name_no_ext = os.path.splitext(os.path.basename(clean_filename))[0]
        for file in os.listdir(question_folder):
            if file.startswith(base_name_no_ext) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                possible_paths.append(os.path.join(question_folder, file))
        
        # Find the first path that exists
        image_path = None
        for path in possible_paths:
            if os.path.isfile(path):
                image_path = path
                print(f"Found matching image: {path}")
                break
        
        if image_path:
            # Extract text from the image
            ocr_text = extract_text_from_image(image_path)
            
            # Replace the image reference with the image + OCR text
            # Use the clean filename in the replacement
            image_reference = f"![{alt_text}]({image_filename})"
            replacement = f"![{alt_text}]({clean_filename})\n\n{ocr_text}\n"
            processed_content = processed_content.replace(image_reference, replacement)
            processed_images += 1
        else:
            print(f"No matching image found for: {clean_filename}")
            # Keep the original image reference but add a note
            image_reference = f"![{alt_text}]({image_filename})"
            replacement = f"![{alt_text}]({clean_filename})\n\n[Image: File not found for OCR processing]\n"
            processed_content = processed_content.replace(image_reference, replacement)
    
    print(f"Processed {processed_images} images with OCR")
    return processed_content


def scrape_question_details(question_url, question_folder, question_id):
    """Scrape question details from URL and save as Markdown"""
    # Check if ProcessedText.md already exists - skip if it does
    processed_markdown_path = os.path.join(question_folder, "ProcessedText.md")
    if os.path.exists(processed_markdown_path):
        with print_lock:
            print(f"Question {question_id} already exists - skipping")
        return True
        
    try:
        with print_lock:
            print(f"Scraping question {question_id}: {question_url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(question_url, headers=headers, timeout=30)
        if response.status_code != 200:
            safe_print(f"Failed to access {question_url}, status code: {response.status_code}")
            return False
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main question content
        question_div = soup.select_one('span[itemprop="mainEntity"] .post.view.open')
        if not question_div:
            print(f"Question content not found in {question_url}")
            return False
        
        # Extract the question title
        title_div = question_div.select_one('.title')
        title = title_div.text.strip() if title_div and hasattr(title_div, 'text') else "Untitled Question"
        
        # Extract question user info
        user_info = extract_user_info(question_div)
        
        # Extract question content
        content_div = question_div.select_one('.body .content .wrap')
        
        # Process question content
        content_markdown = "*No content found*"
        if content_div:
            # First, extract and download all images
            content_html = str(content_div)
            image_mapping = extract_and_download_images(content_html, question_folder)
            
            # Then convert HTML to markdown with proper image handling
            content_markdown = custom_html_to_markdown(content_html, image_mapping)
        
        # Create Markdown file with metadata
        markdown_content = f"# {title}\n\n"
        markdown_content += f"**Posted by:** {user_info['name']} | **Date:** {user_info['date']} | **Votes:** {user_info['votes']}\n\n"
        markdown_content += f"{content_markdown}\n\n"
        
        # Extract comments on the question
        try:
            question_comments = extract_comments_with_improved_image_handling(question_div, question_folder)
            if question_comments:
                markdown_content += f"{question_comments}\n"
        except Exception as comment_error:
            print(f"Error extracting question comments: {str(comment_error)}")
            logger.warning(f"Error extracting question comments for {question_url}: {str(comment_error)}")
        
        # Extract all answers, including accepted ones
        try:
            answers = soup.select('div[itemprop="suggestedAnswer"], div[itemprop="acceptedAnswer"]')
            if answers:
                markdown_content += "## Answers\n\n"
                for i, answer in enumerate(answers, 1):
                    try:
                        if answer is None:
                            print(f"Warning: Answer {i} is None, skipping")
                            continue
                            
                        # Check if this is an accepted answer
                        is_accepted = answer.get('itemprop') == 'acceptedAnswer'
                        answer_header = f"### Answer {i}" + (" (Accepted)" if is_accepted else "")
                        
                        # Extract answer content with improved image handling
                        answer_content = extract_answer_content_with_improved_image_handling(answer, question_folder)
                        
                        markdown_content += f"{answer_header}\n\n{answer_content}\n\n"
                    except Exception as answer_error:
                        print(f"Error extracting answer {i}: {str(answer_error)}")
                        logger.warning(f"Error extracting answer {i} for {question_url}: {str(answer_error)}")
                        markdown_content += f"{answer_header}\n\n*Error extracting answer content: {str(answer_error)}*\n\n"
        except Exception as answers_error:
            print(f"Error extracting answers: {str(answers_error)}")
            logger.error(f"Error extracting answers for {question_url}: {str(answers_error)}")
            markdown_content += "\n## Answers\n\n*Error extracting answers section*\n\n"
        
        # Extract tags
        try:
            tags_div = question_div.select('.inplace-tags .ptag') if hasattr(question_div, 'select') else []
            if tags_div:
                markdown_content += "## Tags\n\n"
                for tag in tags_div:
                    if tag is None or not hasattr(tag, 'text'):
                        continue
                    tag_text = tag.text.strip()
                    markdown_content += f"* {tag_text}\n"
        except Exception as tags_error:
            print(f"Error extracting tags: {str(tags_error)}")
            logger.warning(f"Error extracting tags for {question_url}: {str(tags_error)}")
        
        # Save the regular markdown file
        markdown_path = os.path.join(question_folder, "Text.md")
        # Clean any backslashes in image filenames before saving
        markdown_content = clean_image_references(markdown_content)
        with open(markdown_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        
        # Create the processed version with OCR text
        try:
            logger.info(f"Creating processed markdown with OCR text for {question_url}")
            processed_content = create_processed_text_md(markdown_content, question_folder)
            
            # Save the processed markdown file
            with open(processed_markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(processed_content)
            logger.info(f"Saved processed markdown to {processed_markdown_path}")
        except Exception as ocr_error:
            print(f"Error creating processed markdown: {str(ocr_error)}")
            logger.error(f"Error creating processed markdown for {question_url}: {str(ocr_error)}")
            # Create a simplified processed version without OCR
            with open(processed_markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            logger.info(f"Saved simplified processed markdown to {processed_markdown_path}")
        
        # Save the raw HTML for reference if needed
        html_path = os.path.join(question_folder, "raw.html")
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(response.text)
        
        print(f"Saved question details to {markdown_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error scraping question {question_url}: {str(e)}")
        print(f"Error scraping question {question_url}: {e}")
        
        # Create a minimal markdown file to prevent repeated failures
        try:
            markdown_path = os.path.join(question_folder, "Text.md")
            with open(markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(f"# Failed to scrape question {question_id}\n\nURL: {question_url}\n\nError: {str(e)}")
                
            with open(processed_markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(f"# Failed to scrape question {question_id}\n\nURL: {question_url}\n\nError: {str(e)}")
                
            print(f"Created error placeholder files for {question_id}")
        except Exception as write_error:
            print(f"Failed to create error placeholder files: {str(write_error)}")
            
        return False

def extract_comments_with_improved_image_handling(post_div, question_folder):
    """Extract comments with better image handling"""
    comment_markdown = ""
    
    try:
        if post_div is None:
            print("Warning: Received None post_div in extract_comments_with_improved_image_handling")
            return comment_markdown
            
        comments = post_div.select('.comment-list .comment') if hasattr(post_div, 'select') else []
        
        if comments:
            comment_markdown += "\n#### Comments:\n\n"
            for i, comment in enumerate(comments, 1):
                try:
                    if comment is None:
                        continue
                        
                    # Extract user info
                    user_info = extract_user_info(comment)
                    
                    # Extract comment content
                    content_div = comment.select_one('.magnify') if hasattr(comment, 'select_one') else None
                    if content_div:
                        # First, extract and download all images
                        content_html = str(content_div)
                        image_mapping = extract_and_download_images(content_html, question_folder)
                        
                        # Then convert HTML to markdown with proper image handling
                        content_md = custom_html_to_markdown(content_html, image_mapping)
                        
                        # Include votes in the comment header
                        votes_text = f" | Votes: {user_info['votes']}" if user_info['votes'] != "0" else ""
                        comment_markdown += f"**Comment {i}** by {user_info['name']} ({user_info['date']}{votes_text}):\n"
                        comment_markdown += f"{content_md}\n\n"
                except Exception as e:
                    comment_markdown += f"**Comment {i}** - *Error extracting comment: {str(e)}*\n\n"
                    print(f"Error extracting comment {i}: {str(e)}")
    except Exception as e:
        # If we completely fail to extract comments, add an error note
        print(f"Error extracting comments section: {str(e)}")
        return f"\n#### Comments:\n\n*Error extracting comments: {str(e)}*\n\n"
    
    return comment_markdown

# Update extract_answer_content_with_improved_image_handling with similar error handling
def extract_answer_content_with_improved_image_handling(answer_div, question_folder):
    """Extract answer content with better image handling"""
    try:
        if answer_div is None:
            print("Warning: Received None answer_div in extract_answer_content_with_improved_image_handling")
            return "*No answer content (answer_div is None)*"
            
        # Extract user info
        user_info = extract_user_info(answer_div)
        
        # Extract the content
        content_div = answer_div.select_one('.content .wrap') if hasattr(answer_div, 'select_one') else None
        
        answer_markdown = ""
        if content_div:
            # First, extract and download all images
            content_html = str(content_div)
            image_mapping = extract_and_download_images(content_html, question_folder)
            
            # Then convert HTML to markdown with proper image handling
            answer_markdown = custom_html_to_markdown(content_html, image_mapping)
            
            # Add user info and votes as metadata
            answer_metadata = f"**Posted by:** {user_info['name']} | **Date:** {user_info['date']} | **Votes:** {user_info['votes']}\n\n"
            answer_markdown = answer_metadata + answer_markdown
            
            # Extract comments on this answer
            try:
                comments_markdown = extract_comments_with_improved_image_handling(answer_div, question_folder)
                if comments_markdown:
                    answer_markdown += comments_markdown
            except Exception as e:
                print(f"Error extracting answer comments: {str(e)}")
                answer_markdown += f"\n\n*Error extracting comments: {str(e)}*\n\n"
        else:
            answer_markdown = "*No content found in this answer*"
        
        return answer_markdown
    except Exception as e:
        print(f"Error extracting answer content: {str(e)}")
        return f"*Error extracting answer: {str(e)}*"

# Define a worker function for the thread pool
def process_question(row):
    """Process a single question - suitable for threading"""
    question_id = row['QuestionCount']
    question_url = row['URL']
    
    # Create folder for this question
    base_folder_path = os.path.join(os.path.dirname(csv_path), 'Questions')
    question_folder = os.path.join(base_folder_path, str(question_id))
    
    with print_lock:
        ensure_dir(question_folder)
    
    # Scrape and save question details
    success = scrape_question_details(question_url, question_folder, question_id)
    return question_id, success

def process_questions_from_csv(csv_path, base_folder='Questions', max_workers=4):
    """Process all questions from the CSV file using multiple threads"""
    # Create base Questions folder
    base_folder_path = os.path.join(os.path.dirname(csv_path), base_folder)
    ensure_dir(base_folder_path)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        total_questions = len(df)
        safe_print(f"Loaded {total_questions} questions from CSV")
        
        # Keep track of successful and skipped questions
        successful = 0
        skipped = 0
        failed = 0
        
        # Use ThreadPoolExecutor to process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm for a progress bar
            futures = []
            for _, row in df.iterrows():
                futures.append(executor.submit(process_question, row))
            
            # Process results as they complete
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing questions"):
                question_id, success = future.result()
                if success:
                    if os.path.exists(os.path.join(base_folder_path, str(question_id), "ProcessedText.md")):
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
        
        # Print summary
        safe_print(f"\nSummary:")
        safe_print(f"- Total questions: {total_questions}")
        safe_print(f"- Successfully processed: {successful}")
        safe_print(f"- Skipped (already exist): {skipped}")
        safe_print(f"- Failed: {failed}")
        
    except Exception as e:
        safe_print(f"Error processing CSV file: {e}")
        logger.error(f"Error processing CSV file: {e}")

if __name__ == "__main__":
    # Add command line argument parsing for different modes
    import sys
    
    # Check if we're in fix mode
    if len(sys.argv) > 1 and sys.argv[1] == "--fix-markdown":
        # Fix existing markdown files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        questions_dir = os.path.join(base_dir, "Questions")
        
        if os.path.exists(questions_dir):
            fix_markdown_files(questions_dir)
        else:
            safe_print(f"Questions directory not found at {questions_dir}")
        sys.exit(0)
    
    # Check for thread count argument
    thread_count = 20  # Default thread count
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        thread_count = int(sys.argv[1])
        safe_print(f"Using {thread_count} threads for processing")
    
    # Normal operation - process questions from CSV with multiple threads
    csv_path = os.path.join(os.getcwd(), "biostars_filtered_questions.csv")
    process_questions_from_csv(csv_path, max_workers=thread_count)
    safe_print("Question scraping completed.")
 


