import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from matplotlib.font_manager import FontProperties

# Set styles for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def load_data(file_path):
    """Load the BioStars question data from CSV"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} questions")
    return df

def clean_data(df):
    """Clean and prepare data for analysis"""
    # Convert numeric columns
    numeric_cols = ['QuestionCount', 'Votes', 'Replies', 'Views']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle views column - might be in text format like "1.4K"
    if 'Views' in df.columns:
        df['Views'] = df['Views'].apply(lambda x: 
            float(str(x).replace('K', '')) * 1000 if isinstance(x, str) and 'K' in str(x) 
            else float(str(x).replace(',', '')) if isinstance(x, str) 
            else x)
    
    # Extract tags into list format
    if 'Tags' in df.columns:
        df['TagsList'] = df['Tags'].fillna('').apply(lambda x: [tag.strip() for tag in str(x).split('|') if tag.strip()])
        df['TagCount'] = df['TagsList'].apply(len)
    
    # Convert Updated_Year to numeric if present
    if 'Updated_Year' in df.columns:
        df['Updated_Year'] = pd.to_numeric(df['Updated_Year'], errors='coerce')
    
    return df

def analyze_tags(df):
    """Analyze question tags"""
    # Flatten all tags into a single list
    all_tags = []
    for tags in df['TagsList']:
        all_tags.extend(tags)
    
    # Count tag occurrences
    tag_counts = Counter(all_tags)
    
    # Get top tags
    top_n = 30
    top_tags = dict(tag_counts.most_common(top_n))
    
    return {
        'tag_counts': tag_counts,
        'top_tags': top_tags,
        'total_unique_tags': len(tag_counts)
    }

def analyze_text(df):
    """Analyze question titles for common words and patterns"""
    # Combine all titles
    all_titles = ' '.join(df['Title'].fillna(''))
    
    # Clean text
    # Remove special characters and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', ' ', all_titles.lower())
    
    # Expanded stopwords list to include more unnecessary words
    stopwords = [
        'and', 'the', 'to', 'of', 'in', 'for', 'a', 'on', 'with', 'how', 'can', 'is', 'are', 'i', 'my',
        'from', 'what', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those',
        'am', 'an', 'as', 'at', 'be', 'by', 'do', 'does', 'did', 'has', 'have', 'had', 'its', 'it', 
        'or', 'because', 'if', 'no', 'not', 'such', 'than', 'but', 'into', 'between', 'after', 'before',
        'during', 'without', 'within', 'about', 'questions', 'question', 'help', 'need', 'using', 'use',
        'get', 'getting', 'got', 'make', 'makes', 'making', 'made', 'work', 'working', 'worked',
        'any', 'some', 'many', 'much', 'other', 'another', 'new', 'old', 'same'
    ]
    
    words = cleaned_text.split()
    
    # Filter out stopwords and short words (length < 3)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    top_words = dict(word_counts.most_common(30))
    
    return {
        'word_counts': word_counts,
        'top_words': top_words
    }

def analyze_numerical(df):
    """Analyze numerical fields like Votes, Replies, and Views"""
    numerical_stats = {}
    
    for column in ['Votes', 'Replies', 'Views']:
        if column in df.columns:
            numerical_stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'max': df[column].max(),
                'sum': df[column].sum(),
                'distribution': df[column].value_counts().sort_index().head(10).to_dict()
            }
    
    # Calculate correlations between numerical fields
    numerical_stats['correlations'] = df[['Votes', 'Replies', 'Views']].corr().to_dict() if all(col in df.columns for col in ['Votes', 'Replies', 'Views']) else {}
    
    return numerical_stats

def create_comprehensive_plot(df, tag_analysis, text_analysis, numerical_analysis):
    """Create a professional, publication-ready plot with multiple subplots in one image"""
    # Set professional aesthetics
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Create a large figure with a grid layout - updated for 6 plots instead of 8
    fig = plt.figure(figsize=(20, 18))  # Further reduced height since we're removing another plot
    
    # Use more sophisticated gridspec layout - updated for 6 plots in a 3x2 grid
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], 
                           hspace=0.35, wspace=0.25)
    
    # Define color palette for consistency
    colors = {
        'main': '#1f77b4',       # Primary color for main elements
        'accent': '#ff7f0e',     # Accent color for highlights
        'neutral': '#2ca02c',    # Neutral color for secondary elements
        'correlation': 'viridis' # Colormap for correlation heatmap
    }
    
    # === 1. Top tags horizontal bar chart - Upper left ===
    ax1 = fig.add_subplot(gs[0, 0])
    top_tags_df = pd.DataFrame(list(tag_analysis['top_tags'].items()), 
                               columns=['Tag', 'Count']).sort_values('Count', ascending=False).head(15)
    
    # Fixed barplot - assign hue and set legend=False instead of using palette directly
    bars = sns.barplot(x='Count', y='Tag', hue='Tag', data=top_tags_df, 
                palette=sns.color_palette("Blues_d", len(top_tags_df)), 
                ax=ax1, edgecolor='none', alpha=0.8, legend=False)
    
    # Add data labels to bars
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax1.text(width + 1, p.get_y() + p.get_height()/2, 
                 f'{int(width):,}', ha='left', va='center',
                 fontsize=10, color='#333333')
    
    ax1.set_title('Most Common Tags', fontweight='bold', pad=15)
    ax1.set_xlabel('Number of Questions')
    ax1.set_ylabel('')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # === 2. Enhanced Word Cloud - Upper right ===
    ax2 = fig.add_subplot(gs[0, 1])
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        colormap='viridis',
        max_words=100, 
        contour_width=0,
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=80,
        random_state=42
    ).generate_from_frequencies(text_analysis['word_counts'])
    
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.set_title('Common Terms in Question Titles', fontweight='bold', pad=15)
    ax2.axis('off')
    
    # Add a thin border around the wordcloud
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(0.5)
    
    # === 3. Scatter plot with trend line - Middle left ===
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Add transparencies for density visualization
    scat = ax3.scatter(df['Votes'], df['Replies'], 
                       alpha=0.25, s=30, 
                       c=df['Views'], cmap='plasma',
                       edgecolor='none')
    
    # Add a polynomial fit to show trend
    from scipy.stats import linregress
    from numpy.polynomial.polynomial import polyfit
    
    # Only use visible range for better fit
    x_visible = df['Votes'][(df['Votes'] < df['Votes'].quantile(0.95)) & 
                           (df['Replies'] < df['Replies'].quantile(0.95))]
    y_visible = df['Replies'][(df['Votes'] < df['Votes'].quantile(0.95)) &
                             (df['Replies'] < df['Replies'].quantile(0.95))]
    
    if len(x_visible) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x_visible, y_visible)
        x_line = np.linspace(0, x_visible.max(), 100)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add annotation about correlation
        ax3.text(0.05, 0.95, f'Correlation: {r_value:.2f}', transform=ax3.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.5', 
                                               facecolor='white', alpha=0.8))
    
    ax3.set_title('Relationship Between Votes and Replies', fontweight='bold', pad=15)
    ax3.set_xlabel('Votes')
    ax3.set_ylabel('Replies')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(False)
    
    # Improve colorbar
    cbar = plt.colorbar(scat, ax=ax3, pad=0.02)
    cbar.set_label('Views', rotation=270, labelpad=20)
    cbar.outline.set_visible(False)
    
    # === 4. REMOVED: Views distribution - Middle right ===
    # Instead, place Top words horizontal bar here (previously #6)
    ax4 = fig.add_subplot(gs[1, 1])
    
    top_words_df = pd.DataFrame(
        list(text_analysis['top_words'].items()), 
        columns=['Word', 'Frequency']
    ).sort_values('Frequency', ascending=False).head(12)
    
    # Update barplot - using hue instead of palette directly
    bars = sns.barplot(
        x='Frequency', 
        y='Word', 
        hue='Word',
        data=top_words_df, 
        palette=sns.color_palette("YlOrRd", len(top_words_df)), 
        ax=ax4, 
        edgecolor='none', 
        alpha=0.8,
        legend=False
    )
    
    # Add data labels to bars
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax4.text(width + 5, p.get_y() + p.get_height()/2, 
                f'{int(width):,}', ha='left', va='center', fontsize=9)
    
    ax4.set_title('Most Common Words in Question Titles', fontweight='bold', pad=15)
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='x', linestyle='--', alpha=0.3)
    
    # === 5. Tag count distribution with enhanced visuals - Lower left (previously #5) ===
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Calculate percentage for each tag count
    tag_counts = df['TagCount'].value_counts().sort_index()
    tag_pcts = tag_counts / tag_counts.sum() * 100
    
    # Only include tag counts < 10 for better visualization
    tag_pcts = tag_pcts[tag_pcts.index < 10]
    
    # Create dataframe for plotting
    tag_pcts_df = pd.DataFrame({
        'TagCount': tag_pcts.index, 
        'Percentage': tag_pcts.values
    })
    
    # Update barplot - using hue instead of palette directly
    bars = sns.barplot(
        x='TagCount', 
        y='Percentage', 
        hue='TagCount',
        data=tag_pcts_df, 
        palette=sns.color_palette("Blues_d", len(tag_pcts)), 
        ax=ax5, 
        edgecolor='none', 
        alpha=0.85,
        legend=False
    )
    
    # Add percentage labels
    for i, p in enumerate(bars.patches):
        height = p.get_height()
        ax5.text(p.get_x() + p.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', fontsize=9)
    
    ax5.set_title('Distribution of Tags per Question', fontweight='bold', pad=15)
    ax5.set_xlabel('Number of Tags')
    ax5.set_ylabel('Percentage of Questions (%)')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', linestyle='--', alpha=0.3)
    
    # === 6. Correlation heatmap - Lower right (previously #7) ===
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Get correlation matrix with better labels
    correlation_df = df[['Votes', 'Replies', 'Views']].corr()
    
    # Use a mask to only show lower triangle
    mask = np.zeros_like(correlation_df)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Plot heatmap with improved styling
    sns.heatmap(
        correlation_df, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        ax=ax6,
        linewidths=1,
        mask=mask,
        center=0,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    ax6.set_title('Correlation Between Votes, Replies and Views', fontweight='bold', pad=15)
    
    # === Removed plots 4 (Distribution of Question Views) and 8 (Key Dataset Statistics) ===
    
    # Create title with date information
    plt.suptitle(
        'Analysis of BioStars Questions',
        fontsize=24, 
        y=0.98, 
        fontweight='bold',
        color='#333333'
    )
    
    # Add subtitle with dataset information - include more metadata here
    # plt.figtext(
    #     0.5, 0.955, 
    #     f"Analysis of {len(df):,} questions with {tag_analysis['total_unique_tags']:,} unique tags | " + 
    #     f"Avg. Votes: {numerical_analysis['Votes']['mean']:.1f} | " +
    #     f"Avg. Replies: {numerical_analysis['Replies']['mean']:.1f} | " +
    #     f"Median Views: {numerical_analysis['Views']['median']:.0f}",
    #     ha='center', 
    #     fontsize=14, 
    #     color='#666666'
    # )
    
     
    
    # Adjust layout for the 6-plot figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # Save the figure with high resolution
    output_path = os.path.join(os.path.dirname(__file__), f"biostars_eda_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Professional EDA plot saved to {output_path}")
    
    return fig

def main():
    """Main function to run the EDA process"""
    # Define file path
    file_path = os.path.join(os.path.dirname(__file__), "biostars_all_questions.csv")
    
    # Load and clean data
    df = load_data(file_path)
    df = clean_data(df)
    
    # Analyze data
    tag_analysis = analyze_tags(df)
    print(f"Found {tag_analysis['total_unique_tags']} unique tags")
    print(f"Top 5 tags: {list(tag_analysis['top_tags'].items())[:5]}")
    
    text_analysis = analyze_text(df)
    print(f"Top 5 words in titles: {list(text_analysis['top_words'].items())[:5]}")
    
    numerical_analysis = analyze_numerical(df)
    print(f"Average votes: {numerical_analysis['Votes']['mean']:.2f}")
    print(f"Average replies: {numerical_analysis['Replies']['mean']:.2f}")
    
    # Create and save visualization
    fig = create_comprehensive_plot(df, tag_analysis, text_analysis, numerical_analysis)
    
    # Save key findings as text
    findings_path = os.path.join(os.path.dirname(__file__), f"biostars_eda_findings_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(findings_path, 'w') as f:
        f.write("KEY FINDINGS FROM BIOSTARS QUESTIONS EDA\n")
        f.write("======================================\n\n")
        
        f.write(f"Total questions analyzed: {len(df)}\n")
        f.write(f"Total unique tags: {tag_analysis['total_unique_tags']}\n\n")
        
        f.write("Top 10 tags:\n")
        for tag, count in list(tag_analysis['top_tags'].items())[:10]:
            f.write(f"  - {tag}: {count} questions\n")
        
        f.write("\nNumerical statistics:\n")
        for metric, stats in numerical_analysis.items():
            if metric != 'correlations':
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.2f}\n")
                f.write(f"    Median: {stats['median']}\n")
                f.write(f"    Max: {stats['max']}\n")
        
        f.write("\nCorrelations:\n")
        for var1, values in numerical_analysis['correlations'].items():
            for var2, corr in values.items():
                if var1 != var2:
                    f.write(f"  {var1} vs {var2}: {corr:.3f}\n")
        
        f.write("\nMost common words in question titles:\n")
        for word, count in list(text_analysis['top_words'].items())[:15]:
            f.write(f"  - {word}: {count}\n")
            
    print(f"Key findings saved to {findings_path}")
    
    return df, tag_analysis, text_analysis, numerical_analysis

if __name__ == "__main__":
    main()
 
