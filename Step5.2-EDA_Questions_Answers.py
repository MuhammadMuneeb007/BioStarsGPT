import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
 

# Set publication-quality plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Normalize all tags to lowercase
        for item in data:
            if 'Tags' in item:
                item['Tags'] = [tag.lower() for tag in item['Tags']]
                
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to count tags
def count_tags(data, answer_type=None):
    tag_counts = Counter()
    
    for item in data:
        if answer_type is None or item.get('Answer') == answer_type:
            for tag in item.get('Tags', []):
                tag_counts[tag] += 1
                
    return tag_counts

# Function to calculate text lengths
def calculate_text_lengths(data):
    question_lengths = []
    explanation_lengths = []
    
    for item in data:
        if 'Question1' in item:
            question_lengths.append(len(item['Question1']))
        if 'Explanation' in item:
            explanation_lengths.append(len(item['Explanation']))
            
    return question_lengths, explanation_lengths

# Function to create word cloud from text with improved styling
def create_word_cloud(text, title, output_path, mask=None):
    # Expanded stopword list with more common words to filter out
    stop_words = set(stopwords.words('english'))
    custom_stops = {
        'use', 'using', 'used', 'like', 'want', 'need', 'try', 'tried', 'get', 'getting', 
        'make', 'making', 'also', 'would', 'could', 'should', 'may', 'might', 'must',
        'one', 'two', 'three', 'first', 'second', 'third', 'new', 'old', 'many',
        'much', 'well', 'say', 'says', 'said', 'see', 'sees', 'seen', 'know',
        'knows', 'known', 'thing', 'things', 'way', 'ways', 'time', 'times', 
        'something', 'anything', 'everything', 'someone', 'anyone', 'everyone',
        'somewhere', 'anywhere', 'everywhere', 'etc', 'ie', 'eg'
    }
    stop_words.update(custom_stops)
    
    # More thorough text cleaning
    text = re.sub(r'[^\w\s]', ' ', text.lower())  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)              # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    clean_text = ' '.join(filtered_words)
    
    # Custom colormap for wordcloud - scientific publication style
    colors = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"]
    custom_cmap = LinearSegmentedColormap.from_list("scientific", colors, N=256)
    
    print(clean_text)

    # Create word cloud
    # wordcloud = WordCloud(
    #     width=1200, 
    #     height=800, 
    #     background_color='white',
    #     colormap=custom_cmap,
    #     max_words=150, 
    #     contour_width=0,
    #     prefer_horizontal=0.7,
    #     min_font_size=8,
    #     max_font_size=100,
    #     random_state=42,
    #     collocations=False
    # ).generate(clean_text)
    
    # Create plot
    # plt.figure(figsize=(16, 10))
    # #plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.title(title, fontsize=24, pad=20)
    # plt.tight_layout()
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()

# Function to create a comprehensive question-answer analysis diagram
def create_qa_analysis_diagram(data, viz_dir):
    # Create a larger figure with more subplots to include vocabulary and richness visualizations
    plt.figure(figsize=(24, 24))
    plt.suptitle("Analysis of Questions-Answers", fontsize=28, y=0.98, fontweight='bold')
    
    # Create a grid with 4 rows and 3 columns
    gs = gridspec.GridSpec(4, 3)
    
    # Define a consistent, publication-quality color palette
    yes_color = "#2c7bb6"  # Blue for YES
    no_color = "#d7191c"   # Red for NO
    palette = sns.color_palette("viridis", 10)
    
    # 1. Answer distribution (YES/NO)
    ax1 = plt.subplot(gs[0, 0])
    yes_answers = sum(1 for item in data if item.get('Answer') == 'YES')
    no_answers = sum(1 for item in data if item.get('Answer') == 'NO')
    answer_counts = {'YES': yes_answers, 'NO': no_answers}
    
    sns.barplot(x=list(answer_counts.keys()), y=list(answer_counts.values()), 
               ax=ax1, palette=[yes_color, no_color])
    
    # Add count labels on top of the bars
    for i, count in enumerate(answer_counts.values()):
        ax1.text(i, count + 5, f"{count}", ha='center', fontsize=12)
    
    ax1.set_title('(a) Answer Distribution', fontsize=16, pad=10)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Tag counts for YES answers (Top 10)
    ax2 = plt.subplot(gs[0, 1:])
    yes_tag_counts = count_tags(data, 'YES')
    top_yes_tags = yes_tag_counts.most_common(10)
    top_yes_df = pd.DataFrame(top_yes_tags, columns=['Tag', 'Count'])
    
    # Horizontal bar plot for tags
    bars = sns.barplot(data=top_yes_df, x='Count', y='Tag', palette="viridis", ax=ax2)
    ax2.set_title('(b) Top 10 Tags with YES Answers', fontsize=16, pad=10)
    ax2.set_xlabel('Count', fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add count labels to the end of each bar
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax2.text(width + 1, p.get_y() + p.get_height()/2, f"{int(width)}", 
                ha='left', va='center', fontsize=12)
    
    # 3. Tag counts for NO answers (Top 10)
    ax3 = plt.subplot(gs[1, 0:2])
    no_tag_counts = count_tags(data, 'NO')
    top_no_tags = no_tag_counts.most_common(10)
    top_no_df = pd.DataFrame(top_no_tags, columns=['Tag', 'Count'])
    
    bars = sns.barplot(data=top_no_df, x='Count', y='Tag', palette="rocket", ax=ax3)
    ax3.set_title('(c) Top 10 Tags with NO Answers', fontsize=16, pad=10)
    ax3.set_xlabel('Count', fontsize=14)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add count labels
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax3.text(width + 1, p.get_y() + p.get_height()/2, f"{int(width)}", 
                ha='left', va='center', fontsize=12)
    
    # 4. Additional Analysis - YES/NO ratio by tag
    ax4 = plt.subplot(gs[1, 2])
    common_tags = set(dict(top_yes_tags).keys()) & set(dict(top_no_tags).keys())
    ratios = []
    
    for tag in common_tags:
        yes_count = yes_tag_counts.get(tag, 0)
        no_count = no_tag_counts.get(tag, 0)
        if yes_count > 0:  # Avoid division by zero
            ratios.append((tag, yes_count/no_count))
    
    # Sort by ratio
    ratios.sort(key=lambda x: x[1], reverse=True)
    ratio_df = pd.DataFrame(ratios[:8], columns=['Tag', 'YES/NO Ratio'])
    
    bars = sns.barplot(data=ratio_df, x='YES/NO Ratio', y='Tag', palette="Blues_r", ax=ax4)
    ax4.set_title('(d) Tags by YES/NO Ratio', fontsize=16, pad=10)
    ax4.set_xlabel('Ratio (higher is better)', fontsize=14)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add ratio labels
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax4.text(width + 0.1, p.get_y() + p.get_height()/2, f"{width:.2f}", 
                ha='left', va='center', fontsize=12)
    
    # 5. Question and Explanation Length Distributions
    ax5 = plt.subplot(gs[2, 0])
    question_lengths, explanation_lengths = calculate_text_lengths(data)
    
    # Use KDE plots for smoother distribution visualization
    sns.kdeplot(question_lengths, ax=ax5, fill=True, color=palette[2], alpha=0.7, linewidth=2, label="Question Length")
    sns.kdeplot(explanation_lengths, ax=ax5, fill=True, color=palette[6], alpha=0.7, linewidth=2, label="Explanation Length")
    
    ax5.set_title('(e) Content Length Distributions', fontsize=16, pad=10)
    ax5.set_xlabel('Length (characters)', fontsize=14)
    ax5.set_ylabel('Density', fontsize=14)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.legend()
    
    # 6. Advanced Analysis - Tag Co-occurrence Network (simplified representation)
    ax6 = plt.subplot(gs[2, 1:])
    
    tag_pairs = []
    for item in data:
        tags = item.get('Tags', [])
        if len(tags) > 1:
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    tag_pairs.append(tuple(sorted([tag1, tag2])))
    
    tag_pair_counts = Counter(tag_pairs)
    top_pairs = tag_pair_counts.most_common(15)
    
    pair_names = [f"{t1}\n+ {t2}" for (t1, t2), _ in top_pairs]
    pair_counts = [count for _, count in top_pairs]
    
    bars = sns.barplot(x=pair_names, y=pair_counts, palette="viridis", ax=ax6)
    ax6.set_title('(f) Top 15 Tag Co-occurrences', fontsize=16, pad=10)
    ax6.set_ylabel('Count', fontsize=14)
    ax6.set_xlabel('Tag Pairs', fontsize=14)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    plt.setp(ax6.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add count labels
    for i, p in enumerate(bars.patches):
        height = p.get_height()
        ax6.text(p.get_x() + p.get_width()/2, height + 0.1, f"{int(height)}", 
                ha='center', va='bottom', fontsize=11)
    
    # 7. NEW: Database Content Richness Analysis with radar chart
    ax7 = plt.subplot(gs[3, 0], polar=True)
    
    # Extract questions and explanations for vocabulary analysis
    questions = [item.get('Question1', '') for item in data if 'Question1' in item]
    explanations = [item.get('Explanation', '') for item in data if 'Explanation' in item]
    all_texts = questions + explanations
    
    # Process texts to calculate vocabulary metrics (simplified version)
    stop_words = set(stopwords.words('english'))
    all_words = []
    for text in all_texts:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    total_words = len(all_words)
    unique_words = len(set(all_words))
    
    # Calculate domain-specific metrics
    word_freq = Counter(all_words)
    hapax_count = sum(1 for word, count in word_freq.items() if count == 1)
    hapax_percentage = (hapax_count / unique_words) * 100 if unique_words > 0 else 0
    technical_words = sum(1 for word in set(all_words) if len(word) > 8)
    technical_percentage = (technical_words / unique_words) * 100 if unique_words > 0 else 0
    
    # Calculate simple diversity metrics
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Define categories for radar chart
    categories = [
        'Vocabulary Size', 
        'Lexical Diversity', 
        'Rare Words',
        'Technical Terms', 
        'Content Coverage',
        'Topic Specificity',
        'Writing Complexity'
    ]
    
    # Normalize values between 0 and 1 for radar chart
    richness_scores = [
        min(1.0, unique_words / 15000),        # Vocabulary size (cap at 15k words)
        min(1.0, ttr * 2.5),                  # Lexical diversity
        min(1.0, hapax_percentage / 60),       # Rare words (cap at 60%)
        min(1.0, technical_percentage / 40),   # Technical terms (cap at 40%)
        min(1.0, len(set([tag for item in data for tag in item.get('Tags', [])])) / 150),  # Topic coverage
        min(1.0, len(set([tag for item in data for tag in item.get('Tags', [])]))/ 
            sum(1 for item in data for tag in item.get('Tags', []))),  # Topic specificity
        min(1.0, 0.5 + (sum(len(word) for word in set(all_words)) / len(set(all_words)) - 4) / 10)  # Writing complexity
    ]
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Close the loop
    angles += angles[:1]
    richness_scores += richness_scores[:1]
    
    # Plot with enhanced styling
    ax7.fill(angles, richness_scores, color='#5D9CEC', alpha=0.25)
    ax7.plot(angles, richness_scores, color='#5D9CEC', linewidth=2, linestyle='solid', marker='o', markersize=8)
    
    # Add category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Add radial labels
    ax7.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], 
              ['0.2', '0.4', '0.6', '0.8', '1.0'], 
              color='grey', size=10)
    plt.ylim(0, 1)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add title
    ax7.set_title('(g) Content Richness', fontsize=16, y=1.08)
    
    # Add annotation to explain the radar chart
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    annotation_text = (
        f"Unique words: {unique_words}\n"
        f"Lexical Diversity: {ttr:.3f}\n"
        f"Topic coverage: {len(set([tag for item in data for tag in item.get('Tags', [])]))}"
    )
    plt.figtext(0.18, 0.07, annotation_text, fontsize=10, bbox=props, ha='center')
    
    # 8. REPLACE: Zipf's Law Analysis instead of Vocabulary Distribution
    ax8 = plt.subplot(gs[3, 1:])
    
    # Process texts to extract word frequencies
    # Extract questions and explanations for vocabulary analysis
    questions = [item.get('Question1', '') for item in data if 'Question1' in item]
    explanations = [item.get('Explanation', '') for item in data if 'Explanation' in item]
    all_texts = questions + explanations
    
    # Process texts to calculate vocabulary metrics
    stop_words = set(stopwords.words('english'))
    all_words = []
    for text in all_texts:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    # Calculate word frequencies
    word_freq = Counter(all_words)
    
    # Get the most common words for analysis
    most_common = word_freq.most_common(300)
    ranks = list(range(1, len(most_common) + 1))
    frequencies = [count for _, count in most_common]
    
    # Plot on log-log scale with enhanced styling for publication quality
    ax8.loglog(ranks, frequencies, marker='o', linestyle='none', 
              color='#1f77b4', markersize=4, alpha=0.7, markeredgecolor='white',
              markeredgewidth=0.5, label="Word Frequencies")
    
    # Add Zipf's law reference line
    k = frequencies[0]
    expected = [k/r for r in ranks]
    ax8.loglog(ranks, expected, color='#d62728', linewidth=2, alpha=0.7, 
              linestyle='--', label="Zipf's Law (1/r)")
    
    # Add regression line
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    fit_line = 10 ** (slope * np.log10(ranks) + intercept)
    ax8.loglog(ranks, fit_line, color='#2ca02c', linewidth=2, 
              linestyle='-', label=f"Best Fit (slope={slope:.2f})")
    
    # Calculate R²
    correlation_matrix = np.corrcoef(log_ranks, log_freqs)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    # Add R² as small text in corner
    ax8.text(0.05, 0.05, f"R² = {r_squared:.3f}", transform=ax8.transAxes, 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax8.set_xlabel('Word Rank (log scale)', fontsize=14)
    ax8.set_ylabel('Frequency (log scale)', fontsize=14)
    ax8.set_title("(h) Zipf's Law Analysis", fontsize=16, pad=10)
    ax8.grid(True, alpha=0.3, which='both', linestyle='--')
    ax8.legend(fontsize=12, loc='upper right')
    
    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, wspace=0.3, hspace=0.4)
    
    qa_analysis_path = f"{viz_dir}/question_answer_analysis.png"
    plt.savefig(qa_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive Question-Answer Analysis diagram to {qa_analysis_path}")
    return qa_analysis_path

# Function for advanced content similarity analysis
def analyze_content_similarity(data, viz_dir):
    # Extract questions text
    questions = [item.get('Question1', '') for item in data if 'Question1' in item]
    
    # TF-IDF vectorization for content analysis
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3
    )
    
    try:
        # Only process if we have enough questions
        if len(questions) > 10:
            tfidf_matrix = tfidf_vectorizer.fit_transform(questions[:500])  # Limit to 500 for performance
            
            # Calculate mean similarity
            sample_size = min(100, tfidf_matrix.shape[0])
            indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
            sample_matrix = tfidf_matrix[indices]
            
            similarity_matrix = cosine_similarity(sample_matrix)
            np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity
            
            # Plot similarity heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_matrix, cmap='viridis', 
                      xticklabels=False, yticklabels=False, vmin=0, vmax=1)
            plt.title('Question Content Similarity (Sampled)', fontsize=18, pad=20)
            plt.tight_layout()
            sim_map_path = f"{viz_dir}/question_similarity_heatmap.png"
            plt.savefig(sim_map_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot similarity distribution
            sim_values = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
            
            plt.figure(figsize=(12, 6))
            sns.histplot(sim_values, kde=True, bins=50, color="#2c7bb6", alpha=0.7)
            plt.title('Distribution of Question Similarities', fontsize=18, pad=20)
            plt.xlabel('Cosine Similarity', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            sim_dist_path = f"{viz_dir}/similarity_distribution.png"
            plt.savefig(sim_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created content similarity analysis visuals")
            return {
                "similarity_heatmap": sim_map_path,
                "similarity_distribution": sim_dist_path,
                "avg_similarity": float(np.mean(sim_values)),
                "median_similarity": float(np.median(sim_values))
            }
    except Exception as e:
        print(f"Error in content similarity analysis: {e}")
    
    return {}

# Analyze answer length vs. response type
def analyze_answer_relationship(data, viz_dir):
    if not data:
        return {}
    
    # Extract data
    lengths = []
    responses = []
    
    for item in data:
        if 'Question1' in item and 'Answer' in item:
            answer = item['Answer']
            # Fix: Only include YES and NO answers
            if answer == 'YES' or answer == 'NO':
                lengths.append(len(item['Question1']))
                responses.append(answer)
    
    # Create dataframe
    df = pd.DataFrame({'Length': lengths, 'Response': responses})
    
    # Check if we have enough data to create the plots
    if df.empty or len(df['Response'].unique()) < 2:
        print("Not enough data to create answer relationship visualizations.")
        return {}
    
    # Create violin plot comparing distributions
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Response', y='Length', data=df, palette={"YES": "#2c7bb6", "NO": "#d7191c"}, 
                  inner='quartile', cut=0)
    plt.title('Question Length by Answer Type', fontsize=18, pad=20)
    plt.xlabel('Answer Type', fontsize=14)
    plt.ylabel('Question Length (characters)', fontsize=14)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    qa_length_path = f"{viz_dir}/question_length_by_answer.png"
    plt.savefig(qa_length_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create length comparison boxplot
    plt.figure(figsize=(14, 6))
    
    # Create a custom color palette - blue for YES, red for NO
    custom_palette = {"YES": "#2c7bb6", "NO": "#d7191c"}
    
    # Draw boxplots with individual points visible
    sns.boxplot(x='Response', y='Length', data=df, palette=custom_palette, width=0.5)
    sns.stripplot(x='Response', y='Length', data=df, 
                 palette=custom_palette, alpha=0.3, 
                 jitter=True, size=3, dodge=True)
    
    plt.title('Question Length Distribution by Answer', fontsize=18, pad=20)
    plt.ylabel('Question Length (characters)', fontsize=14)
    plt.xlabel('Answer Type', fontsize=14)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    qa_box_path = f"{viz_dir}/question_length_boxplot.png"
    plt.savefig(qa_box_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created answer relationship analysis visuals")
    return {
        "length_violin_plot": qa_length_path,
        "length_boxplot": qa_box_path
    }

# New function to analyze database uniqueness and richness
def analyze_database_uniqueness(data, viz_dir):
    """
    Analyze the richness and uniqueness of the database content
    by examining vocabulary uniqueness, content diversity, and topic coverage.
    """
    print("\nAnalyzing database uniqueness and richness...")
    
    # Extract questions and explanations
    questions = [item.get('Question1', '') for item in data if 'Question1' in item]
    explanations = [item.get('Explanation', '') for item in data if 'Explanation' in item]
    all_texts = questions + explanations
    
    if not all_texts:
        print("No text content found for analysis.")
        return {}
    
    # 1. Enhanced Vocabulary richness analysis
    print("Analyzing vocabulary richness...")
    
    # Preprocessing function for text with improved handling
    def preprocess_text(text):
        # Enhanced stopword list
        stop_words = set(stopwords.words('english'))
        additional_stopwords = {
            'use', 'using', 'used', 'like', 'want', 'need', 'try', 'tried', 'get', 'getting', 
            'make', 'making', 'also', 'would', 'could', 'should', 'may', 'might', 'must',
            'one', 'two', 'three', 'first', 'second', 'third', 'new', 'old', 'many',
            'much', 'well', 'say', 'says', 'said', 'see', 'sees', 'seen', 'know',
            'knows', 'known', 'thing', 'things', 'way', 'ways', 'time', 'times', 
            'something', 'anything', 'everything', 'someone', 'anyone', 'everyone',
            'somewhere', 'anywhere', 'everywhere', 'etc', 'ie', 'eg',
            'however', 'therefore', 'thus', 'hence', 'although', 'though',
            'despite', 'whereas', 'while', 'since', 'because', 'cant', 'doesnt',
            'dont', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt',
            'wouldnt', 'couldnt', 'shouldnt', 'cant', 'cannot', 'wont'
        }
        stop_words.update(additional_stopwords)
        
        # More thorough text cleaning
        # Remove special chars and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove digits and common number formats
        text = re.sub(r'\b\d+\b|\b\d+[.,]\d+\b', ' ', text)
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|\S+@\S+|\S+\.com\S*|\S+\.org\S*|\S+\.net\S*', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into words
        words = text.split()
        # Remove stopwords and very short words
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words
    
    # Process all texts and build vocabulary with more detailed metrics
    all_words = []
    question_words = []
    explanation_words = []
    
    for text in questions:
        processed = preprocess_text(text)
        all_words.extend(processed)
        question_words.extend(processed)
        
    for text in explanations:
        processed = preprocess_text(text)
        all_words.extend(processed)
        explanation_words.extend(processed)
    
    # Basic vocabulary metrics - enhanced
    total_words = len(all_words)
    unique_words = len(set(all_words))
    
    # Calculate metrics for questions and explanations separately
    q_total_words = len(question_words)
    q_unique_words = len(set(question_words))
    e_total_words = len(explanation_words)
    e_unique_words = len(set(explanation_words))
    
    # Calculate shared vocabulary between questions and explanations
    shared_vocab = set(question_words).intersection(set(explanation_words))
    shared_vocab_size = len(shared_vocab)
    
    # Calculate unique vocabulary in each category
    q_exclusive_vocab = set(question_words).difference(set(explanation_words))
    e_exclusive_vocab = set(explanation_words).difference(set(question_words))
    
    q_exclusive_size = len(q_exclusive_vocab)
    e_exclusive_size = len(e_exclusive_vocab)
    
    if total_words == 0:
        print("No words found after preprocessing.")
        return {}
    
    # Enhanced lexical diversity metrics
    # Type-Token Ratio (measure of lexical diversity)
    ttr = unique_words / total_words
    
    # Root TTR (scales better with text length)
    rttr = unique_words / np.sqrt(total_words)
    
    # Corrected TTR using Moving Average TTR (MATTR) approximation
    # This helps address the issue of TTR being sensitive to text length
    window_size = min(1000, total_words // 2) if total_words > 2000 else total_words // 2
    if window_size > 10:  # Only calculate if we have enough text
        windows = max(1, (total_words - window_size) // (window_size // 2))
        mattr_values = []
        for i in range(windows):
            start = i * (window_size // 2)
            end = start + window_size
            if end <= total_words:
                window_words = all_words[start:end]
                window_ttr = len(set(window_words)) / len(window_words)
                mattr_values.append(window_ttr)
        
        mattr = np.mean(mattr_values) if mattr_values else ttr
    else:
        mattr = ttr
    
    # Calculate frequency distribution
    word_freq = Counter(all_words)
    
    # Hapax legomena (words that appear only once)
    hapax_count = sum(1 for word, count in word_freq.items() if count == 1)
    hapax_percentage = (hapax_count / unique_words) * 100
    
    # Dis legomena (words that appear twice)
    dis_count = sum(1 for word, count in word_freq.items() if count == 2)
    dis_percentage = (dis_count / unique_words) * 100
    
    # Calculate Yule's K (measure of lexical richness - lower values indicate more richness)
    m1 = total_words
    m2 = sum([freq ** 2 for word, freq in word_freq.items()])
    yules_k = (m2 - m1) / (m1 * m1) * 10000 if m1 > 0 else 0
    
    # Technical vocabulary (approximation using word length as proxy)
    technical_words = sum(1 for word in set(all_words) if len(word) > 8)
    technical_percentage = (technical_words / unique_words) * 100
    
    # Domain specificity metric - ratio of rare/technical words to common words
    domain_specificity = technical_words / (unique_words - technical_words) if (unique_words - technical_words) > 0 else 0
    
    # Print enhanced metrics
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Type-Token Ratio: {ttr:.4f}")
    print(f"Root TTR: {rttr:.4f}")
    print(f"Moving Average TTR: {mattr:.4f}")
    print(f"Yule's K measure: {yules_k:.2f}")
    print(f"Hapax legomena: {hapax_count} ({hapax_percentage:.2f}% of vocabulary)")
    print(f"Dis legomena: {dis_count} ({dis_percentage:.2f}% of vocabulary)")
    print(f"Technical vocabulary: {technical_words} ({technical_percentage:.2f}% of vocabulary)")
    print(f"Domain specificity ratio: {domain_specificity:.4f}")
    print(f"Shared vocabulary between questions and explanations: {shared_vocab_size} words")
    print(f"Question-exclusive vocabulary: {q_exclusive_size} words")
    print(f"Explanation-exclusive vocabulary: {e_exclusive_size} words")
    
    # Generate a comprehensive vocabulary statistics table
    vocabulary_stats = pd.DataFrame({
        'Metric': [
            'Total Words', 'Unique Words', 'Type-Token Ratio', 
            'Root TTR', 'Moving Average TTR', 'Yule\'s K',
            'Hapax Legomena', 'Hapax %', 'Dis Legomena', 'Dis %',
            'Technical Words', 'Technical %', 'Domain Specificity'
        ],
        'Value': [
            total_words, unique_words, ttr,
            rttr, mattr, yules_k,
            hapax_count, hapax_percentage, dis_count, dis_percentage,
            technical_words, technical_percentage, domain_specificity
        ]
    })
    
    # Create a publication-quality table visualization
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=vocabulary_stats.values,
        colLabels=vocabulary_stats.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Add title
    plt.suptitle('Vocabulary Statistics', fontsize=20, y=0.95)
    
    # Save table
    vocab_stats_path = f"{viz_dir}/vocabulary_statistics.png"
    plt.savefig(vocab_stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced Content diversity visualization
    print("Analyzing content diversity...")
    
    # Create a more comprehensive radar chart for vocabulary richness metrics
    categories = [
        'Vocabulary Size', 
        'Lexical Diversity', 
        'Rare Words',
        'Technical Terms', 
        'Content Coverage',
        'Domain Specificity',
        'Writing Complexity'
    ]
    
    # Normalize values between 0 and 1 for radar chart using improved metrics
    # These are relative scores - higher is better
    richness_scores = [
        min(1.0, unique_words / 15000),        # Vocabulary size (cap at 15k words)
        min(1.0, mattr * 2.5),                 # Lexical diversity using MATTR (scaled)
        min(1.0, hapax_percentage / 60),       # Rare words (cap at 60%)
        min(1.0, technical_percentage / 40),   # Technical terms (cap at 40%)
        min(1.0, len(set([tag for item in data for tag in item.get('Tags', [])])) / 150),  # Topic coverage
        min(1.0, domain_specificity * 2),      # Domain specificity (scaled)
        min(1.0, (1 - yules_k / 200) if yules_k < 200 else 0.1)  # Writing complexity (inverse of Yule's K)
    ]
    
    # Create an enhanced radar chart with better styling
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Close the loop
    angles += angles[:1]
    richness_scores += richness_scores[:1]
    
    # Plot with enhanced styling
    ax.fill(angles, richness_scores, color='#5D9CEC', alpha=0.25)
    ax.plot(angles, richness_scores, color='#5D9CEC', linewidth=2, linestyle='solid', marker='o', markersize=8)
    
    # Add category labels with better positioning
    plt.xticks(angles[:-1], categories, size=14)
    
    # Add radial labels with improved styling
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], 
              ['0.2', '0.4', '0.6', '0.8', '1.0'], 
              color='grey', size=12)
    plt.ylim(0, 1)
    
    # Add grid with styling
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add title with enhanced styling
    plt.title('Database Content Richness Analysis', size=22, y=1.1, fontweight='bold')
    
    # Add annotation to explain the radar chart metrics
    annotation_text = (
        f"Vocabulary: {unique_words} unique words from {total_words} total\n"
        f"Lexical Diversity: {mattr:.3f} (MATTR)\n"
        f"Rare Words: {hapax_percentage:.1f}% appear only once\n"
        f"Technical Terms: {technical_percentage:.1f}% of vocabulary\n"
        f"Domain Specificity: {domain_specificity:.3f}\n"
        f"Content Coverage: {len(set([tag for item in data for tag in item.get('Tags', [])]))} topics\n"
        f"Writing Complexity: {yules_k:.2f} (Yule's K)"
    )
    
    # Create a text box for annotations
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.figtext(0.15, 0.15, annotation_text, fontsize=12, 
               bbox=props, verticalalignment='bottom')
    
    # Save the enhanced radar chart
    richness_radar_path = f"{viz_dir}/database_richness_radar.png"
    plt.savefig(richness_radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced Word frequency distribution (Zipf's law check) - Improved with publication quality
    plt.figure(figsize=(14, 10))
    
    # Get the most common words for analysis
    most_common = word_freq.most_common(300)  # Analyze more words for better accuracy
    ranks = list(range(1, len(most_common) + 1))
    frequencies = [count for _, count in most_common]
    words = [word for word, _ in most_common]
    
    # Create a custom colormap for a more professional look
    main_color = '#1f77b4'  # A professional blue color
    
    # Plot on log-log scale with enhanced styling for publication quality
    plt.loglog(ranks, frequencies, marker='o', linestyle='none', 
              color=main_color, markersize=6, alpha=0.7, markeredgecolor='white',
              markeredgewidth=0.5, label="Observed Word Frequencies")
    
    # Add Zipf's law reference line with improved explanation
    k = frequencies[0]  # Frequency of the most common word
    expected = [k/r for r in ranks]
    plt.loglog(ranks, expected, color='#d62728', linewidth=2.5, alpha=0.7, 
              linestyle='--', label="Zipf's Law Reference (1/r)")
    
    # Add regression line to measure how well it fits Zipf's law
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    
    # Fit line to log-transformed data
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    fit_line = 10 ** (slope * np.log10(ranks) + intercept)
    
    plt.loglog(ranks, fit_line, color='#2ca02c', linewidth=2.5, 
              linestyle='-', label=f"Best Fit: slope={slope:.2f}")
    
    # Calculate R² to measure goodness of fit
    correlation_matrix = np.corrcoef(log_ranks, log_freqs)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    # Enhanced styling for publication quality
    plt.xlabel('Word Rank (log scale)', fontsize=16, fontweight='bold')
    plt.ylabel('Word Frequency (log scale)', fontsize=16, fontweight='bold')
    plt.title("Zipf's Law Analysis of Vocabulary Distribution", fontsize=20, pad=20, fontweight='bold')
    
    # Add grid with professional styling
    plt.grid(True, alpha=0.3, which='both', linestyle='--', color='#cccccc')
    
    # Add legend with professional styling
    legend = plt.legend(fontsize=14, loc='upper right', frameon=True, 
                      fancybox=True, framealpha=0.95, edgecolor='#cccccc')
    
    # Adjust tick parameters for publication quality
    plt.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=7)
    plt.tick_params(axis='both', which='minor', width=1, length=4)
    
    # Add R² value to corner (not on top)
    plt.figtext(0.17, 0.23, f"R² = {r_squared:.3f}", fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    # Show top 5 most frequent words in bottom corner instead of top
    top_words_text = "Top 5 words:\n" + "\n".join([f"{i+1}. {word} ({freq})" 
                                               for i, (word, freq) in enumerate(most_common[:5])])
    plt.figtext(0.72, 0.15, top_words_text, fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the enhanced plot with high resolution for publication
    zipf_path = f"{viz_dir}/vocabulary_zipf_analysis.png"
    plt.savefig(zipf_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    # 4. Enhanced vocabulary composition analysis - new visualization
    # Analyze word length distribution as proxy for complexity
    word_lengths = [len(word) for word in set(all_words)]
    
    plt.figure(figsize=(14, 8))
    
    # Create word length histogram with KDE
    sns.histplot(word_lengths, kde=True, bins=range(1, max(word_lengths) + 1), 
               color='#3498db', alpha=0.7)
    
    # Add mean and median annotations
    mean_length = np.mean(word_lengths)
    median_length = np.median(word_lengths)
    
    plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_length:.2f}')
    plt.axvline(median_length, color='green', linestyle=':', linewidth=2, 
               label=f'Median: {median_length:.2f}')
    
    plt.xlabel('Word Length (characters)', fontsize=14)
    plt.ylabel('Number of Unique Words', fontsize=14)
    plt.title('Word Length Distribution - Content Complexity Indicator', fontsize=18, pad=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Improve x-axis
    plt.xticks(range(1, max(word_lengths) + 1, 2))
    
    word_length_path = f"{viz_dir}/word_length_distribution.png"
    plt.savefig(word_length_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. NEW: Vocabulary overlap between questions and explanations - custom implementation
    # Replace venn2 function with custom circles
    plt.figure(figsize=(12, 10))
    
    # Calculate percentages for overlap visualization
    if shared_vocab_size + q_exclusive_size > 0:
        q_shared_pct = shared_vocab_size / (shared_vocab_size + q_exclusive_size) * 100
    else:
        q_shared_pct = 0
        
    if shared_vocab_size + e_exclusive_size > 0:
        e_shared_pct = shared_vocab_size / (shared_vocab_size + e_exclusive_size) * 100
    else:
        e_shared_pct = 0
    
    # Create a custom overlap visualization using matplotlib patches
    from matplotlib.patches import Circle, Ellipse
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Determine circle sizes based on vocabulary sizes
    total_size = q_exclusive_size + e_exclusive_size + shared_vocab_size
    q_size = q_exclusive_size + shared_vocab_size
    e_size = e_exclusive_size + shared_vocab_size
    
    # Scale the circles based on their relative sizes
    q_radius = np.sqrt(q_size / total_size) * 3
    e_radius = np.sqrt(e_size / total_size) * 3
    
    # Position the circles with appropriate overlap
    overlap = shared_vocab_size / total_size * 2
    distance = q_radius + e_radius - overlap
    
    # Create and add the circles
    q_circle = Circle((-distance/2, 0), q_radius, alpha=0.5, fc='#3498db', ec='black')
    e_circle = Circle((distance/2, 0), e_radius, alpha=0.5, fc='#e74c3c', ec='black')
    
    ax.add_patch(q_circle)
    ax.add_patch(e_circle)
    
    # Add labels for the sections
    ax.text(-distance, 0, f"Questions Only\n{q_exclusive_size} words", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(distance, 0, f"Explanations Only\n{e_exclusive_size} words", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, 0, f"Shared\n{shared_vocab_size} words", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Set axis limits and remove ticks
    ax.set_xlim(-q_radius - 1, e_radius + 1)
    ax.set_ylim(-max(q_radius, e_radius) - 1, max(q_radius, e_radius) + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add title
    plt.title('Vocabulary Overlap Between Questions and Explanations', fontsize=18, pad=20)
    
    # Add legend
    q_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                          markersize=15, label='Questions Vocabulary')
    e_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
                          markersize=15, label='Explanations Vocabulary')
    plt.legend(handles=[q_legend, e_legend], loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14)
    
    vocab_overlap_path = f"{viz_dir}/vocabulary_overlap.png"
    plt.savefig(vocab_overlap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Topic uniqueness analysis - visualizing tag relationships (using existing code)
    tag_counts = Counter([tag for item in data for tag in item.get('Tags', [])])
    unique_tags = set(tag_counts.keys())
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    tag_matrix = np.zeros((len(unique_tags), len(unique_tags)))
    
    for item in data:
        tags = item.get('Tags', [])
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                idx1 = tag_to_idx[tag1]
                idx2 = tag_to_idx[tag2]
                tag_matrix[idx1][idx2] += 1
                tag_matrix[idx2][idx1] += 1
    
    tag_uniqueness = {}
    for tag in unique_tags:
        idx = tag_to_idx[tag]
        co_occurrences = np.sum(tag_matrix[idx] > 0)
        uniqueness_score = 1 - (co_occurrences / (len(unique_tags) - 1))
        tag_uniqueness[tag] = uniqueness_score
    
    unique_topics = sorted([(tag, score) for tag, score in tag_uniqueness.items()], 
                          key=lambda x: x[1], reverse=True)
    
    top_unique_tags = unique_topics[:20]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if top_unique_tags:
        tags, scores = zip(*top_unique_tags)
        colors = plt.cm.viridis(np.array(scores))
        y_pos = np.arange(len(tags))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tags, fontsize=12)
        ax.set_xlabel('Uniqueness Score', fontsize=14)
        ax.set_title('Top 20 Unique Topics in the Database', fontsize=18, pad=15)
        
        for i, tag in enumerate(tags):
            count = tag_counts[tag]
            ax.text(scores[i]+0.01, i, f"({count} occurrences)", va='center', fontsize=10)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                 norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Uniqueness Score', fontsize=12)
    else:
        ax.text(0.5, 0.5, "No unique topics found", 
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    unique_topics_path = f"{viz_dir}/unique_topics.png"
    plt.savefig(unique_topics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "vocabulary_metrics": {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": ttr,
            "moving_average_ttr": mattr,
            "root_ttr": rttr,
            "yules_k": yules_k,
            "hapax_percentage": hapax_percentage,
            "dis_legomena_percentage": dis_percentage,
            "technical_vocabulary_percentage": technical_percentage,
            "domain_specificity": domain_specificity,
            "average_word_length": mean_length,
            "zipf_slope": slope,
            "zipf_rsquared": r_squared
        },
        "vocabulary_comparison": {
            "question_vocabulary": q_unique_words,
            "explanation_vocabulary": e_unique_words,
            "shared_vocabulary": shared_vocab_size,
            "question_exclusive": q_exclusive_size,
            "explanation_exclusive": e_exclusive_size,
            "question_shared_percentage": q_shared_pct,
            "explanation_shared_percentage": e_shared_pct
        },
        "unique_topics_count": len(unique_tags),
        "richness_visualizations": {
            "richness_radar": richness_radar_path,
            "zipf_analysis": zipf_path,
            "word_length_distribution": word_length_path,
            "vocabulary_statistics": vocab_stats_path,
            "vocabulary_overlap": vocab_overlap_path,
            "unique_topics": unique_topics_path
        }
    }

# Main analysis function
def analyze_questions(file_path):
    # Load the data
    data = load_data(file_path)
    if not data:
        return
    
    print("Analyzing questions...")
    
    # 1. Count the total number of questions
    total_questions = len(data)
    print(f"Total number of questions: {total_questions}")
    
    # 2. Count the total number of Answer "Yes", and "No"
    yes_answers = sum(1 for item in data if item.get('Answer') == 'YES')
    no_answers = sum(1 for item in data if item.get('Answer') == 'NO')
    print(f"Total 'YES' answers: {yes_answers}")
    print(f"Total 'NO' answers: {no_answers}")
    
    # 3. Count the tags with high number of Yes
    yes_tag_counts = count_tags(data, 'YES')
    top_yes_tags = yes_tag_counts.most_common(20)
    print("\nTop 20 tags with 'YES' answers:")
    for tag, count in top_yes_tags:
        print(f"{tag}: {count}")
    
    # 4. Count the tags with high number of No
    no_tag_counts = count_tags(data, 'NO')
    top_no_tags = no_tag_counts.most_common(20)
    print("\nTop 20 tags with 'NO' answers:")
    for tag, count in top_no_tags:
        print(f"{tag}: {count}")
    
    # 4.1 Find tags with high NO/YES ratio (hard to answer or unanswered)
    print("\nTags that are difficult to answer (high NO/YES ratio):")
    min_occurrences = 5  # Minimum number of occurrences for both YES and NO
    hard_tags = {}
    
    for tag in set(yes_tag_counts.keys()) | set(no_tag_counts.keys()):
        yes_count = yes_tag_counts.get(tag, 0)
        no_count = no_tag_counts.get(tag, 0)
        
        if yes_count >= min_occurrences and no_count >= min_occurrences:
            ratio = no_count / yes_count if yes_count > 0 else float('inf')
            hard_tags[tag] = (ratio, yes_count, no_count)
    
    hard_tags_sorted = sorted(hard_tags.items(), key=lambda x: x[1][0], reverse=True)
    for tag, (ratio, yes_count, no_count) in hard_tags_sorted[:20]:
        print(f"{tag}: NO/YES ratio = {ratio:.2f} (YES: {yes_count}, NO: {no_count})")
    
    # Count average question and answer length
    question_lengths, explanation_lengths = calculate_text_lengths(data)
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    avg_explanation_length = sum(explanation_lengths) / len(explanation_lengths) if explanation_lengths else 0
    
    print(f"\nAverage question length: {avg_question_length:.2f} characters")
    print(f"Average explanation length: {avg_explanation_length:.2f} characters")
    
    # Data visualization
    print("\nGenerating visualizations...")
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "/data/ascher02/uqmmune1/BioStarsGPT/visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Generate comprehensive question-answer analysis diagram
    qa_analysis_path = create_qa_analysis_diagram(data, viz_dir)
    
    # Perform additional analyses
    similarity_info = analyze_content_similarity(data, viz_dir)
    answer_relationship_info = analyze_answer_relationship(data, viz_dir)
    
    # NEW: Analyze database uniqueness and richness
    uniqueness_info = analyze_database_uniqueness(data, viz_dir)
    
    # Plot high-quality distribution of answers
    plt.figure(figsize=(10, 6))
    answer_counts = {'YES': yes_answers, 'NO': no_answers}
    
    bars = sns.barplot(x=list(answer_counts.keys()), y=list(answer_counts.values()), 
                      palette=["#2c7bb6", "#d7191c"])
    
    # Add count labels on top of the bars
    for i, count in enumerate(answer_counts.values()):
        plt.text(i, count + 5, f"{count}", ha='center', fontsize=14)
        
    plt.title('Distribution of Answers', fontsize=18, pad=15)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    answer_dist_path = f"{viz_dir}/answer_distribution.png"
    plt.savefig(answer_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top tags with YES answers - publication quality
    plt.figure(figsize=(14, 10))
    top_yes_df = pd.DataFrame(top_yes_tags, columns=['Tag', 'Count'])
    bars = sns.barplot(data=top_yes_df, x='Count', y='Tag', palette='viridis')
    
    # Add count labels
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2, f"{int(width)}", 
                ha='left', va='center', fontsize=12)
                
    plt.title('Top 20 Tags with YES Answers', fontsize=18, pad=15)
    plt.xlabel('Count', fontsize=16)
    plt.ylabel('Tag', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    yes_tags_path = f"{viz_dir}/top_yes_tags.png"
    plt.savefig(yes_tags_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top tags with NO answers - publication quality
    plt.figure(figsize=(14, 10))
    top_no_df = pd.DataFrame(top_no_tags, columns=['Tag', 'Count'])
    bars = sns.barplot(data=top_no_df, x='Count', y='Tag', palette='rocket')
    
    # Add count labels
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2, f"{int(width)}", 
                ha='left', va='center', fontsize=12)
                
    plt.title('Top 20 Tags with NO Answers', fontsize=18, pad=15)
    plt.xlabel('Count', fontsize=16)
    plt.ylabel('Tag', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    no_tags_path = f"{viz_dir}/top_no_tags.png"
    plt.savefig(no_tags_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot publication-quality length distributions
    plt.figure(figsize=(14, 8))
    
    # Use KDE plots for smoother visualization
    sns.kdeplot(question_lengths, color="#2c7bb6", fill=True, alpha=0.5, 
               label="Question Lengths", linewidth=2, cut=0)
    plt.axvline(x=avg_question_length, color="#2c7bb6", linestyle='--', linewidth=2)
    
    plt.title('Distribution of Question Lengths', fontsize=18, pad=15)
    plt.xlabel('Length (characters)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    q_len_path = f"{viz_dir}/question_length_distribution.png"
    plt.savefig(q_len_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot for explanation lengths
    plt.figure(figsize=(14, 8))
    
    sns.kdeplot(explanation_lengths, color="#d7191c", fill=True, alpha=0.5, 
               label="Explanation Lengths", linewidth=2, cut=0)
    plt.axvline(x=avg_explanation_length, color="#d7191c", linestyle='--', linewidth=2)
    
    plt.title('Distribution of Explanation Lengths', fontsize=18, pad=15)
    plt.xlabel('Length (characters)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    e_len_path = f"{viz_dir}/explanation_length_distribution.png"
    plt.savefig(e_len_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create enhanced word clouds
    all_questions = ' '.join([item.get('Question1', '') for item in data])
    all_explanations = ' '.join([item.get('Explanation', '') for item in data])
    
    create_word_cloud(all_questions, 'Common Terms in Questions', f"{viz_dir}/question_wordcloud.png")
    create_word_cloud(all_explanations, 'Common Terms in Explanations', f"{viz_dir}/explanation_wordcloud.png")
    
    # Create separate word clouds for YES and NO answers
    yes_questions = ' '.join([item.get('Question1', '') for item in data if item.get('Answer') == 'YES'])
    no_questions = ' '.join([item.get('Question1', '') for item in data if item.get('Answer') == 'NO'])
    
    create_word_cloud(yes_questions, 'Terms in Questions with YES Answers', f"{viz_dir}/yes_question_wordcloud.png")
    create_word_cloud(no_questions, 'Terms in Questions with NO Answers', f"{viz_dir}/no_question_wordcloud.png")
    
    # 5 & 6. Save questions as Train.json and Test.json with proper paths
    train_data = []
    test_data = []
    
    for item in data:
        if item.get('Answer') == 'YES':
            tags = item.get('Tags', [])
            if len(tags) > 1:
                train_data.append(item)
            elif len(tags) == 1:
                test_data.append(item)
 
    eda_path = "/data/ascher02/uqmmune1/BioStarsGPT/EDA.json"
     
    
    # Save all analysis results to EDA.json
    eda_results = {
        'total_questions': total_questions,
        'yes_answers': yes_answers,
        'no_answers': no_answers,
        'top_yes_tags': top_yes_tags,
        'top_no_tags': top_no_tags,
        'hard_to_answer_tags': [(tag, ratio, yes, no) for tag, (ratio, yes, no) in hard_tags_sorted[:20]],
        'avg_question_length': avg_question_length,
        'avg_explanation_length': avg_explanation_length,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'visualization_paths': {
            'comprehensive_analysis': qa_analysis_path,
            'answer_distribution': answer_dist_path,
            'yes_tags': yes_tags_path,
            'no_tags': no_tags_path,
            'question_length': q_len_path,
            'explanation_length': e_len_path,
            'question_wordcloud': f"{viz_dir}/question_wordcloud.png",
            'explanation_wordcloud': f"{viz_dir}/explanation_wordcloud.png",
            'yes_question_wordcloud': f"{viz_dir}/yes_question_wordcloud.png",
            'no_question_wordcloud': f"{viz_dir}/no_question_wordcloud.png"
        }
    }
    
    # Add additional analysis results
    eda_results.update(similarity_info)
    eda_results.update(answer_relationship_info)
    eda_results.update({"database_uniqueness": uniqueness_info})
    
    with open(eda_path, 'w', encoding='utf-8') as f:
        json.dump(eda_results, f, indent=2)
    
    print(f"\nSaved analysis results to {eda_path}")
    print("Analysis complete!")

if __name__ == "__main__":
    analyze_questions("Allquestions.json")
