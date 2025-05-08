import json
import os
import numpy as np
import textstat
from transformers import GPT2Tokenizer
import spacy
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

# Read Allquestions.json
def read_json_file(file_path):
    """Read JSON file and return the data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def filterbasedonyesorno(data):
    """Filter entries based on 'Answer' field - keep only 'YES' answers."""
    if not data:
        return [], 0, 0
    
    kept_entries = []
    discard_count = 0
    
    for entry in tqdm(data, desc="Filtering YES/NO"):
        if "Answer" in entry and entry["Answer"] == "YES":
            kept_entries.append(entry)
        else:
            discard_count += 1
    
    kept_count = len(kept_entries)
    print(f"Discarded entries: {discard_count}")
    print(f"Kept entries: {kept_count}")
    
    return kept_entries, kept_count, discard_count

def filterbasedoncontext(data):
    """Filter entries based on token count in explanations - keep only those with < 1048 tokens in both explanations."""
    if not data:
        return [], 0, 0
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    kept_entries = []
    discard_count = 0
    
    for entry in tqdm(data, desc="Filtering by token count"):
        # Check both Explanation1 and Explanation2
        explanation1_ok = True
        explanation2_ok = True
        
        if "Explanation1" in entry:
            token_count1 = len(tokenizer.encode(entry["Explanation1"]))
            explanation1_ok = token_count1 < 1048
        
        if "Explanation2" in entry:
            token_count2 = len(tokenizer.encode(entry["Explanation2"]))
            explanation2_ok = token_count2 < 1048
        
        # Keep entry only if both explanations are within token limit
        if explanation1_ok and explanation2_ok:
            kept_entries.append(entry)
        else:
            discard_count += 1
    
    kept_count = len(kept_entries)
    print(f"Entries discarded due to token count: {discard_count}")
    print(f"Entries kept after token filtering: {kept_count}")
    
    return kept_entries, kept_count, discard_count

def process_explanation(entry, explanation_key, nlp):
    """Process a single explanation and return complexity features."""
    if explanation_key not in entry or not entry[explanation_key]:
        return None, None

    text = entry[explanation_key]
    
    # Skip if text is too short for meaningful analysis
    if len(text) < 50:
        return None, None
    
    # Initialize feature dictionary
    feature_dict = {}
    
    # 1. Readability scores (fast metrics only)
    feature_dict["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
    feature_dict["gunning_fog"] = textstat.gunning_fog(text)
    feature_dict["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
    feature_dict["automated_readability_index"] = textstat.automated_readability_index(text)
    
    # 2. Basic text statistics
    feature_dict["char_count"] = textstat.char_count(text, ignore_spaces=True)
    feature_dict["lexicon_count"] = textstat.lexicon_count(text, removepunct=True)
    feature_dict["sentence_count"] = textstat.sentence_count(text)
    
    # 3. Simplified syntactic analysis with spaCy
    doc = nlp(text)
    
    # POS tag diversity (simplified)
    pos_counts = Counter([token.pos_ for token in doc])
    feature_dict["pos_diversity"] = len(pos_counts) / len(doc) if len(doc) > 0 else 0
    
    # 4. Lexical sophistication (simplified)
    # Lexical diversity (Type-Token Ratio)
    words = [token.text.lower() for token in doc if token.is_alpha]
    feature_dict["type_token_ratio"] = len(set(words)) / len(words) if words else 0
    
    # 5. Sentence complexity
    feature_dict["avg_sentence_length"] = feature_dict["lexicon_count"] / feature_dict["sentence_count"] if feature_dict["sentence_count"] > 0 else 0
    
    # Calculate simplified complexity score
    complexity_score = (
        # Readability (lower is more complex) - normalized and inverted
        0.25 * min(max((100 - feature_dict["flesch_reading_ease"]) / 100, 0), 1) +
        0.20 * min(feature_dict["gunning_fog"] / 18, 1) +
        # Lexical sophistication
        0.25 * feature_dict["type_token_ratio"] +
        # Sentence complexity
        0.30 * min(feature_dict["avg_sentence_length"] / 30, 1)
    )
    
    # Classify complexity
    if complexity_score < 0.35:
        complexity = "Low"
    elif complexity_score < 0.55:
        complexity = "Medium"
    elif complexity_score < 0.75:
        complexity = "High"
    else:
        complexity = "Very High"
    
    # Create feature vector for clustering
    feature_vector = [
        # Readability
        min(max((100 - feature_dict["flesch_reading_ease"]) / 100, 0), 1),
        min(feature_dict["gunning_fog"] / 18, 1),
        min(feature_dict["flesch_kincaid_grade"] / 16, 1),
        # Lexical
        feature_dict["type_token_ratio"],
        # Sentence
        min(feature_dict["avg_sentence_length"] / 30, 1)
    ]
    
    # Return results
    results = {
        "complexity_features": {k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) else v 
                            for k, v in feature_dict.items()},
        "complexity": complexity,
        "complexity_score": float(complexity_score)
    }
    
    return results, feature_vector

def calculate_complexity_parallel(data, batch_size=1000):
    """Calculate complexity of explanations using parallel processing for speed."""
    if not data:
        return [], {}, []
    
    print("Loading spaCy model...")
    # Use smaller, faster model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # For very large pipes, disable components we don't need
    nlp.disable_pipes("ner")
    
    # Complexity categories
    complexity_categories = {"Low": 0, "Medium": 0, "High": 0, "Very High": 0}
    explanation_count = 0
    
    # For storing all feature vectors for clustering
    all_features = []
    all_entries_with_explanations = []
    all_scores = []
    
    # Number of CPU cores to use
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        # Process all explanations in this batch
        batch_results = []
        for entry in tqdm(batch, desc="Analyzing complexity"):
            entry_results = {}
            
            # Process both explanations
            for explanation_key in ["Explanation1", "Explanation2"]:
                results, feature_vector = process_explanation(entry, explanation_key, nlp)
                
                if results and feature_vector:
                    # Store results
                    entry[explanation_key + "_complexity_features"] = results["complexity_features"]
                    entry[explanation_key + "_complexity"] = results["complexity"]
                    entry[explanation_key + "_complexity_score"] = results["complexity_score"]
                    
                    # Track for statistics
                    complexity_categories[results["complexity"]] += 1
                    explanation_count += 1
                    
                    # Store for clustering
                    all_features.append(feature_vector)
                    all_entries_with_explanations.append((entry, explanation_key))
                    all_scores.append(results["complexity_score"])
        
    # Calculate percentages
    complexity_stats = {
        "counts": complexity_categories,
        "percentages": {k: (v / explanation_count * 100) for k, v in complexity_categories.items() if explanation_count > 0},
        "total_explanations": explanation_count
    }
    
    return data, complexity_stats, (all_features, all_entries_with_explanations, all_scores)

def cluster_by_publication_level(feature_data, max_samples=10000):
    """Cluster explanations by publication level based on complexity features."""
    print("\nClustering explanations by publication level...")
    
    all_features, all_entries_with_explanations, all_scores = feature_data
    
    if not all_features:
        print("No features available for clustering.")
        return {}
    
    # Sample if too many entries to reduce computation time
    if len(all_features) > max_samples:
        print(f"Sampling {max_samples} out of {len(all_features)} explanations for clustering...")
        indices = np.random.choice(len(all_features), max_samples, replace=False)
        sampled_features = [all_features[i] for i in indices]
        sampled_entries = [all_entries_with_explanations[i] for i in indices]
        sampled_scores = [all_scores[i] for i in indices]
    else:
        sampled_features = all_features
        sampled_entries = all_entries_with_explanations
        sampled_scores = all_scores
    
    # Convert to numpy arrays
    features_array = np.array(sampled_features)
    scores_array = np.array(sampled_scores)
    
    # Define publication clusters
    n_clusters = 5
    
    # Apply K-Means clustering
    print("Applying K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_array)
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Assign publication names based on average complexity score of each cluster
    cluster_avg_scores = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_scores = scores_array[cluster_indices]
        cluster_avg_scores[i] = np.mean(cluster_scores)
    
    # Sort clusters by average complexity score
    sorted_clusters = sorted(cluster_avg_scores.items(), key=lambda x: x[1])
    
    # Map clusters to publication levels
    publication_levels = [
        "Elementary / Popular Blog",  # Lowest complexity
        "High School / General Magazine",
        "Undergraduate / News Magazine",
        "Graduate / Specialized Journal", 
        "Expert / Scientific Journal"   # Highest complexity
    ]
    
    cluster_to_publication = {}
    for i, (cluster_id, _) in enumerate(sorted_clusters):
        cluster_to_publication[cluster_id] = publication_levels[i]
    
    # Now predict publication levels for ALL entries
    print("Assigning publication levels to all entries...")
    
    # Function to predict publication level for single feature vector
    def predict_publication_level(features):
        cluster_id = kmeans.predict([features])[0]
        return cluster_to_publication[cluster_id]
    
    # Assign publication levels to entries in batches
    publication_counts = {level: 0 for level in publication_levels}
    
    # For sampled entries, we already have the labels
    for i, (entry, explanation_key) in enumerate(sampled_entries):
        cluster_id = cluster_labels[i]
        publication = cluster_to_publication[cluster_id]
        entry[explanation_key + "_publication_level"] = publication
        publication_counts[publication] += 1
    
    # For non-sampled entries, predict labels
    if len(all_features) > max_samples:
        non_sampled_indices = set(range(len(all_features))) - set(indices)
        for i in tqdm(non_sampled_indices, desc="Predicting publication levels"):
            feature_vector = all_features[i]
            entry, explanation_key = all_entries_with_explanations[i]
            publication = predict_publication_level(feature_vector)
            entry[explanation_key + "_publication_level"] = publication
            publication_counts[publication] += 1
    
    # Generate statistics
    total = sum(publication_counts.values())
    publication_stats = {
        "counts": publication_counts,
        "percentages": {k: (v / total * 100) for k, v in publication_counts.items()},
        "total_explanations": total
    }
    
    # Create simplified visualization with sampled data
    create_publication_cluster_visualization(sampled_features, cluster_labels, cluster_to_publication, cluster_centers)
    
    return publication_stats

def create_publication_cluster_visualization(features, labels, cluster_to_publication, centers):
    """Create a publication-quality visualization of the publication clusters."""
    print("Creating publication-quality visualization...")
    
    # Use PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Calculate variance explained by the principal components
    explained_variance = pca.explained_variance_ratio_
    
    # Sample if too many points (for better visualization)
    max_points = min(5000, len(features))
    if len(features) > max_points:
        indices = np.random.choice(len(features), max_points, replace=False)
        features_2d_sample = features_2d[indices]
        labels_sample = labels[indices]
    else:
        features_2d_sample = features_2d
        labels_sample = labels
    
    # Create high-quality figure with specified style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    
    # Better colormap - using a colorblind-friendly palette
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Professional color palette that's distinguishable
    if num_clusters <= 5:
        # For 5 or fewer clusters, use this specific color palette
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    else:
        # For more clusters, use a viridis-based palette
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_clusters))
    
    # Plot each cluster with enhanced styling
    handles = []
    for i, label in enumerate(unique_labels):
        mask = labels_sample == label
        scatter = ax.scatter(
            features_2d_sample[mask, 0],
            features_2d_sample[mask, 1],
            s=40, 
            alpha=0.7,
            color=colors[i],
            edgecolors='none',
            marker='o',
            label=f"{cluster_to_publication[label]}"
        )
        handles.append(scatter)
    
    # Plot cluster centers with enhanced marker style
    centers_2d = pca.transform(centers)
    ax.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        s=200, 
        marker='X', 
        color='black',
        edgecolors='white',
        linewidth=1.5,
        zorder=10,
        label="Cluster Centers"
    )
    
    # Add text labels for cluster centers
    for i, (x, y) in enumerate(centers_2d):
        publication = cluster_to_publication[i]
        # Create shortened label for cleaner display
        short_label = publication.split('/')[0].strip()
        ax.annotate(
            short_label,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color='black',
            backgroundcolor='white',
            alpha=0.8,
            zorder=11
        )
    
    # Add axis labels with variance explained
    ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}% variance)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}% variance)", 
                 fontsize=14, fontweight='bold')
    
    # Add title with styling
    ax.set_title("Explanation Complexity Clusters by Publication Level", 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add detailed legend with custom position and styling
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12
    )
    
    # Add grid with subtle styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Add border to the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Add text box with information about the plot
    info_text = (
        f"PCA dimensionality reduction of complexity features\n"
        f"Total explanations: {len(features)}\n"
        f"Sampled points: {len(features_2d_sample)}"
    )
    
    # Add the info box
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Tight layout for better spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure in high resolution
    output_path = "publication_clusters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Publication-quality visualization saved as {output_path}")
    
    # Also save as vector graphic for publication purposes
    vector_path = "publication_clusters.pdf"
    plt.savefig(vector_path, format='pdf', bbox_inches='tight')
    print(f"Vector graphic saved as {vector_path}")

def main():
    # Path to your JSON file
    file_path = "Allquestions.json"
    
    # Read the JSON file
    print(f"Reading {file_path}...")
    all_data = read_json_file(file_path)
    if not all_data:
        return
    
    print(f"Total entries loaded: {len(all_data)}")
    
    # Apply filters sequentially
    print("\n--- Filtering based on YES/NO answers ---")
    yes_filtered_data, yes_kept, yes_discarded = filterbasedonyesorno(all_data)
    
    print("\n--- Filtering based on explanation token count ---")
    token_filtered_data, token_kept, token_discarded = filterbasedoncontext(yes_filtered_data)
    
    # Save intermediate results after filtering to avoid reprocessing
    intermediate_output = "filtered_questions_intermediate.json"
    print(f"Saving intermediate filtered data to {intermediate_output}")
    with open(intermediate_output, 'w', encoding='utf-8') as f:
        json.dump(token_filtered_data, f, ensure_ascii=False, indent=2)
    
    print("\n--- Calculating complexity of explanations ---")
    complexity_data, complexity_stats, feature_data = calculate_complexity_parallel(token_filtered_data)
    
    # Print complexity statistics
    print("\n=== Explanation Complexity Statistics ===")
    print(f"Total explanations analyzed: {complexity_stats['total_explanations']}")
    print("\nComplexity distribution:")
    for category, count in complexity_stats["counts"].items():
        percentage = complexity_stats["percentages"][category]
        print(f"  {category}: {count} explanations ({percentage:.2f}%)")
    
    # Save another intermediate result
    complexity_output = "filtered_questions_with_complexity.json"
    print(f"Saving complexity data to {complexity_output}")
    with open(complexity_output, 'w', encoding='utf-8') as f:
        json.dump(complexity_data, f, ensure_ascii=False, indent=2)
    
    # Perform publication clustering
    publication_stats = cluster_by_publication_level(feature_data)
    
    # Print publication statistics
    print("\n=== Publication Level Distribution ===")
    print(f"Total explanations clustered: {publication_stats['total_explanations']}")
    print("\nDistribution by publication level:")
    for publication, count in publication_stats["counts"].items():
        percentage = publication_stats["percentages"][publication]
        print(f"  {publication}: {count} explanations ({percentage:.2f}%)")
    
    # Save the final data with complexity and publication information
    output_path = "filtered_questions_with_publication_clusters.json"
    print(f"Saving final data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(complexity_data, f, ensure_ascii=False, indent=2)
    
    # Save publication statistics separately
    stats_output_path = "publication_cluster_statistics.json"
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(publication_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Original entry count: {len(all_data)}")
    print(f"Final entry count: {len(complexity_data)}")

if __name__ == "__main__":
    main()