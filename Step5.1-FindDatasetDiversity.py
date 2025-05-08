import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import seaborn as sns
from tqdm import tqdm
import os
from collections import Counter
import time
from wordcloud import WordCloud
import pandas as pd
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA

# Set publication-quality plot settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.figsize'] = (10, 8)
plt.style.use('seaborn-v0_8-whitegrid')

def load_json_data(file_path):
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def create_embeddings(texts, max_features=1500):
    """Create TF-IDF embeddings for a list of texts"""
    print(f"Creating embeddings for {len(texts)} texts...")
    
    # Use TF-IDF vectorizer with max_features to keep dimensionality manageable
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85
    )
    
    # Fit and transform the texts to get embeddings
    try:
        embeddings = vectorizer.fit_transform(texts)
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings, vectorizer
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None, None

def extract_explanations(data):
    """Extract both explanation texts together from JSON data"""
    all_explanations = []
    explanation_sources = []  # To track if text is from Explanation1, Explanation2 or both
    valid_indices = []
    
    print("Extracting both explanations from data...")
    
    for i, item in enumerate(data):
        exp1 = item.get('Explanation1', '')
        exp2 = item.get('Explanation2', '')
        
        # If both explanations exist, combine them for a more comprehensive analysis
        if exp1 and exp2:
            combined_text = exp1 + " " + exp2
            all_explanations.append(combined_text)
            explanation_sources.append("Both")
            valid_indices.append(i)
        elif exp1:
            all_explanations.append(exp1)
            explanation_sources.append("Explanation1")
            valid_indices.append(i)
        elif exp2:
            all_explanations.append(exp2)
            explanation_sources.append("Explanation2")
            valid_indices.append(i)
    
    stats = {
        'total_items': len(data),
        'items_with_explanation1': sum(1 for item in data if item.get('Explanation1', '')),
        'items_with_explanation2': sum(1 for item in data if item.get('Explanation2', '')),
        'items_with_both': sum(1 for item in data if item.get('Explanation1', '') and item.get('Explanation2', '')),
        'explanation_sources': Counter(explanation_sources)
    }
    
    print(f"Found {stats['items_with_explanation1']} items with Explanation1")
    print(f"Found {stats['items_with_explanation2']} items with Explanation2")
    print(f"Found {stats['items_with_both']} items with both explanations")
    print(f"Created {len(all_explanations)} combined explanation texts")
    
    return all_explanations, explanation_sources, valid_indices, stats

def find_optimal_clusters(embeddings, max_clusters=15):
    """Find optimal number of clusters using silhouette score"""
    print("Determining optimal number of clusters...")
    
    # Convert sparse to dense if needed
    if hasattr(embeddings, "toarray"):
        dense_embeddings = embeddings.toarray()
    else:
        dense_embeddings = embeddings
    
    # Sample data if there's too much
    max_samples = 10000
    if dense_embeddings.shape[0] > max_samples:
        indices = np.random.choice(dense_embeddings.shape[0], max_samples, replace=False)
        sample_data = dense_embeddings[indices]
    else:
        sample_data = dense_embeddings
    
    # Try different numbers of clusters
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, sample_data.shape[0]))
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sample_data)
        silhouette_avg = silhouette_score(sample_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
    
    # Find the best number of clusters
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(cluster_range), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Create the ClusterAnalysis directory if it doesn't exist
    os.makedirs('ClusterAnalysis', exist_ok=True)
    
    plt.savefig('ClusterAnalysis/optimal_clusters.png')
    print("Saved optimal clusters plot to ClusterAnalysis/optimal_clusters.png")
    
    return optimal_clusters

def cluster_embeddings(embeddings, n_clusters):
    """Cluster the embeddings using KMeans"""
    print(f"Clustering embeddings into {n_clusters} clusters...")
    
    # Convert sparse matrix to dense if needed
    if hasattr(embeddings, "toarray"):
        dense_embeddings = embeddings.toarray()
    else:
        dense_embeddings = embeddings
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dense_embeddings)
    
    # Calculate cluster statistics
    cluster_counts = Counter(clusters)
    print("Cluster distribution:")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster_id}: {count} questions ({count/len(clusters)*100:.1f}%)")
    
    return clusters, kmeans

def reduce_dimensions(embeddings, n_components=2):
    """Reduce dimensions for visualization using t-SNE"""
    print(f"Reducing dimensions with t-SNE...")
    start_time = time.time()
    
    # Convert sparse matrix to dense if needed
    if hasattr(embeddings, "toarray"):
        dense_embeddings = embeddings.toarray()
    else:
        dense_embeddings = embeddings
    
    # Use PCA first to reduce dimensionality for faster t-SNE processing
    # This allows us to process much larger datasets
    print("Applying PCA to reduce initial dimensionality before t-SNE...")
    pca = PCA(n_components=min(50, dense_embeddings.shape[1]))
    pca_result = pca.fit_transform(dense_embeddings)
    print(f"PCA reduced dimensions from {dense_embeddings.shape[1]} to {pca_result.shape[1]}")
    
    # Apply t-SNE on all data points after PCA reduction
    tsne = TSNE(
        n_components=n_components, 
        random_state=42, 
        perplexity=min(40, len(pca_result) // 100 + 10),  # Scale perplexity with dataset size
        learning_rate='auto', 
        init='random',
        n_iter=1000,
        verbose=1  # Show progress
    )
    print(f"Applying t-SNE on all {len(pca_result)} points...")
    reduced = tsne.fit_transform(pca_result)
    indices = np.arange(dense_embeddings.shape[0])
    
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    return reduced, indices

def get_cluster_labels(texts, clusters, n_clusters, vectorizer=None):
    """Get the most representative terms for each cluster"""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            feature_names = vectorizer.get_feature_names()
    else:
        X = vectorizer.transform(texts)
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            feature_names = vectorizer.get_feature_names()
    
    # For each cluster, find the top terms
    cluster_labels = {}
    
    for i in range(n_clusters):
        # Get indices of texts in this cluster
        indices = [j for j, c in enumerate(clusters) if c == i]
        
        if not indices:
            cluster_labels[i] = f"Cluster {i}"
            continue
            
        # Get the average tfidf values for this cluster
        cluster_values = X[indices].toarray().mean(axis=0)
        
        # Get the top terms for this cluster
        top_indices = cluster_values.argsort()[-5:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        
        # Create a short readable label from the top terms
        if len(top_terms) > 0:
            # Take the top 2 most important terms
            label = " / ".join(top_terms[:2])
        else:
            label = f"Cluster {i}"
            
        cluster_labels[i] = label
    
    return cluster_labels

def create_publication_plot(reduced_embeddings, clusters, explanation_sources, cluster_labels, output_file="ClusterAnalysis/explanation_clusters_publication.png"):
    """Create a high-quality visualization of the clusters with labels"""
    # Create a larger figure for publication quality
    plt.figure(figsize=(16, 12))
    
    # Create a palette for the clusters - use a different color scheme
    unique_clusters = sorted(list(set(clusters)))
    n_clusters = len(unique_clusters)
    
    # Choose a more distinct color palette based on the number of clusters
    if n_clusters <= 10:
        palette = sns.color_palette("tab10", n_clusters)  # Distinct colors for fewer clusters
    elif n_clusters <= 20:
        palette = sns.color_palette("tab20", n_clusters)  # Tab20 has 20 distinct colors
    else:
        # For many clusters, use a combination of color schemes
        palette1 = sns.color_palette("tab20", 20)
        palette2 = sns.color_palette("husl", n_clusters - 20)
        palette = palette1 + palette2[:n_clusters-20]
    
    # Create a DataFrame for easier plotting with seaborn
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': [str(c) for c in clusters],
        'source': explanation_sources
    })
    
    # Get cluster sizes
    cluster_sizes = Counter(clusters)
    
    # Plot each cluster with its own color
    ax = plt.subplot(111)
    
    # First plot a semi-transparent background for all points
    plt.scatter(
        df['x'], 
        df['y'],
        alpha=0.1,
        s=30,  # Smaller point size for better handling of large datasets
        c='gray',
        edgecolor=None
    )
    
    # Then plot each cluster with its own color
    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = df[df['cluster'] == str(cluster_id)]
        plt.scatter(
            cluster_points['x'], 
            cluster_points['y'],
            alpha=0.7,  # Slightly less opacity for better visibility when dense
            s=50,
            c=[palette[i]],
            label=f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} items",
            edgecolor='w',
            linewidth=0.2
        )
    
    # Add cluster annotations with count and label
    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = df[df['cluster'] == str(cluster_id)]
        centroid_x = cluster_points['x'].mean()
        centroid_y = cluster_points['y'].mean()
        
        # Calculate standard deviation to determine label position
        std_x = cluster_points['x'].std() * 0.3
        std_y = cluster_points['y'].std() * 0.3
        
        # Create text for cluster size and top terms
        count = cluster_sizes[cluster_id]
        size_text = f"Cluster {cluster_id}\n{count} items"
        terms_text = f"{cluster_labels[cluster_id]}"
        
        # Add text label for cluster ID and size with colored box matching cluster color
        txt1 = plt.text(
            centroid_x, centroid_y + std_y, 
            size_text,
            fontsize=12,
            weight='bold',
            ha='center',
            va='bottom',
            color='black',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=palette[i], boxstyle='round,pad=0.5', linewidth=2)
        )
        
        # Add text label for top terms with outline
        txt2 = plt.text(
            centroid_x, centroid_y - std_y, 
            terms_text,
            fontsize=11,
            ha='center',
            va='top',
            style='italic'
        )
        txt2.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white')
        ])
    
    # Draw edges between nearby clusters to show relationships
    from scipy.spatial.distance import pdist, squareform
    
    # Get cluster centroids
    centroids = {}
    for cluster_id in unique_clusters:
        cluster_points = df[df['cluster'] == str(cluster_id)]
        centroids[cluster_id] = (cluster_points['x'].mean(), cluster_points['y'].mean())
    
    # Calculate distances between centroids
    centroid_points = np.array(list(centroids.values()))
    distances = squareform(pdist(centroid_points))
    
    # Connect nearby clusters with faint lines
    threshold = np.percentile(distances[distances > 0], 30)  # Connect closest 30% of clusters
    
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            if distances[i, j] < threshold:
                plt.plot(
                    [centroid_points[i, 0], centroid_points[j, 0]],
                    [centroid_points[i, 1], centroid_points[j, 1]],
                    'k-', alpha=0.15, linewidth=1
                )
    
    # Improve plot aesthetics
    plt.title('Clustering of Question Answers in BiostarGPT', fontsize=22, pad=20, weight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=16, labelpad=10)
    plt.ylabel('t-SNE Dimension 2', fontsize=16, labelpad=10)
    
    # Add a subtle grid
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Remove the axis ticks since they don't have meaningful values
    plt.xticks([])
    plt.yticks([])
    
    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Add dataset size info directly on the plot
    dataset_info = f"Dataset: {len(df)} answers | {n_clusters} clusters"
    plt.figtext(0.02, 0.02, dataset_info, ha="left", fontsize=12, weight='bold')
    
    # Improve the legend - create a compact legend if many clusters
    handles, labels = ax.get_legend_handles_labels()
    
    if n_clusters > 10:
        # For many clusters, use a smaller font and more columns
        ncols = min(4, (n_clusters + 4) // 5)  # Scale columns based on number of clusters
        legend = plt.legend(
            handles, labels,
            title="Clusters",
            loc='best',
            frameon=True,
            framealpha=0.95,
            fontsize=8,
            title_fontsize=10,
            ncol=ncols,
            markerscale=0.8
        )
    else:
        # For fewer clusters, use a more standard legend
        legend = plt.legend(
            handles, labels,
            title="Clusters",
            loc='lower right',
            frameon=True,
            framealpha=0.95,
            fontsize=10,
            title_fontsize=12,
            ncol=2  # Two columns for better layout
        )
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Improve layout
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure with high resolution
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Publication-quality plot saved as '{output_file}'")
    
    # Create a second version with a white background for publications that prefer it
    plt.savefig(output_file.replace('.png', '_white_bg.png'), dpi=400, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"White background version saved as '{output_file.replace('.png', '_white_bg.png')}'")
    
    return df

def analyze_clusters_in_detail(texts, clusters, n_clusters, vectorizer, explanation_sources, valid_indices, data):
    """Create detailed analysis for each cluster"""
    os.makedirs('ClusterAnalysis', exist_ok=True)
    
    print("Creating detailed cluster analysis...")
    
    # Create a summary report
    with open('ClusterAnalysis/cluster_summary.txt', 'w') as f:
        f.write("CLUSTER ANALYSIS SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write(f"Total items analyzed: {len(texts)}\n")
        f.write(f"Number of clusters: {n_clusters}\n\n")
        
        # Write distribution of explanation sources
        source_counts = Counter(explanation_sources)
        f.write("Distribution of explanation sources:\n")
        for source, count in source_counts.items():
            f.write(f"  {source}: {count} ({count/len(texts)*100:.1f}%)\n")
        
        f.write("\nCluster distribution:\n")
        cluster_counts = Counter(clusters)
        for cluster_id, count in sorted(cluster_counts.items()):
            f.write(f"  Cluster {cluster_id}: {count} items ({count/len(clusters)*100:.1f}%)\n")
    
    # Analyze each cluster in detail
    cluster_docs = {}
    cluster_metadata = {}
    
    for cluster_id in range(n_clusters):
        # Get indices of texts in this cluster
        indices = [j for j, c in enumerate(clusters) if c == cluster_id]
        
        if not indices:
            continue
            
        # Get the texts for this cluster
        cluster_texts = [texts[j] for j in indices]
        
        # Get sources for this cluster
        cluster_sources = [explanation_sources[j] for j in indices]
        source_distribution = Counter(cluster_sources)
        
        # Create metadata for this cluster
        cluster_metadata[cluster_id] = {
            'size': len(indices),
            'percentage': len(indices)/len(texts)*100,
            'source_distribution': source_distribution,
        }
        
        # Save all texts for this cluster
        cluster_docs[cluster_id] = cluster_texts
        
        # Generate a word cloud for this cluster
        try:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                collocations=False
            ).generate(' '.join(cluster_texts))
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Cluster {cluster_id} Word Cloud', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'ClusterAnalysis/cluster_{cluster_id}_wordcloud.png', dpi=200)
            plt.close()
        except Exception as e:
            print(f"Could not generate word cloud for cluster {cluster_id}: {e}")
        
        # Get example documents from this cluster
        orig_indices = [valid_indices[j] for j in indices if j < len(valid_indices)]
        sample_size = min(5, len(orig_indices))
        if sample_size > 0:
            sample_indices = np.random.choice(orig_indices, sample_size, replace=False)
            
            # Save examples to text file
            with open(f'ClusterAnalysis/cluster_{cluster_id}_examples.txt', 'w') as f:
                f.write(f"CLUSTER {cluster_id} EXAMPLES\n")
                f.write("====================\n\n")
                
                for i, idx in enumerate(sample_indices):
                    if idx < len(data):
                        item = data[idx]
                        f.write(f"Example {i+1}:\n")
                        f.write(f"Question: {item.get('Question', '')[:500]}...\n\n")
                        f.write(f"Explanation1: {item.get('Explanation1', '')[:500]}...\n\n")
                        if item.get('Explanation2'):
                            f.write(f"Explanation2: {item.get('Explanation2', '')[:500]}...\n\n")
                        f.write("-" * 50 + "\n\n")
    
    # Extract top terms for each cluster using the vectorizer
    X = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    with open('ClusterAnalysis/cluster_terms.txt', 'w') as f:
        f.write("CLUSTER TOP TERMS\n")
        f.write("================\n\n")
        
        for i in range(n_clusters):
            indices = [j for j, c in enumerate(clusters) if c == i]
            
            if not indices:
                continue
                
            # Get the average tfidf values for this cluster
            cluster_values = X[indices].toarray().mean(axis=0)
            
            # Get the top terms for this cluster
            top_indices = cluster_values.argsort()[-30:][::-1]  # Get top 30 terms
            top_terms = [(feature_names[idx], cluster_values[idx]) for idx in top_indices]
            
            f.write(f"Cluster {i} ({len(indices)} documents, {len(indices)/len(texts)*100:.1f}%):\n")
            for term, score in top_terms:
                f.write(f"  {term}: {score:.4f}\n")
            f.write("\n")
    
    print("Detailed cluster analysis completed and saved to ClusterAnalysis directory")
    
    return cluster_metadata

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs('ClusterAnalysis', exist_ok=True)
    
    # Configuration parameters to control execution
    MAX_DATAPOINTS = -1  # Set to -1 to use all available data, or to a number to limit (e.g., 10000)
    FORCE_CLUSTERS = None  # Force a specific number of clusters (set to None to use automatic detection)
    
    # Load the data
    file_path = "Allquestions.json"
    data = load_json_data(file_path)
    
    if not data:
        print(f"No data found or could not parse {file_path}")
        exit(1)
    
    print(f"Loaded {len(data)} items from {file_path}")
    
    # Make sure we use more colors for the clusters
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    
    # Extract explanations from the data (combining Explanation1 and Explanation2)
    all_explanations, explanation_sources, valid_indices, stats = extract_explanations(data)
    
    # Apply data limit if specified
    if MAX_DATAPOINTS > 0 and len(all_explanations) > MAX_DATAPOINTS:
        print(f"Limiting analysis to {MAX_DATAPOINTS} data points as configured")
        # Use random sampling to get representative subset
        indices = np.random.choice(len(all_explanations), MAX_DATAPOINTS, replace=False)
        all_explanations = [all_explanations[i] for i in indices]
        explanation_sources = [explanation_sources[i] for i in indices]
        valid_indices = [valid_indices[i] for i in indices]
    
    print(f"Processing {len(all_explanations)} explanations...")
    
    # Create embeddings for the texts
    embeddings, vectorizer = create_embeddings(all_explanations)
    if embeddings is None:
        print("Failed to create embeddings. Exiting.")
        exit(1)
    
    # Determine number of clusters
    if FORCE_CLUSTERS is not None:
        n_clusters = FORCE_CLUSTERS
        print(f"Using forced cluster count: {n_clusters}")
    else:
        # Find optimal number of clusters automatically
        n_clusters = find_optimal_clusters(embeddings)
    
    # Cluster the embeddings
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    # Reduce dimensions for visualization
    reduced_embeddings, sample_indices = reduce_dimensions(embeddings)
    
    # Map clusters to the sampled indices
    sampled_clusters = [clusters[i] for i in sample_indices]
    sampled_sources = [explanation_sources[i] for i in sample_indices]
    
    # Get descriptive labels for each cluster
    sampled_texts = [all_explanations[i] for i in sample_indices]
    cluster_labels = get_cluster_labels(sampled_texts, sampled_clusters, n_clusters, vectorizer)
    
    # Create publication quality plot
    df = create_publication_plot(reduced_embeddings, sampled_clusters, sampled_sources, cluster_labels)
    
    # Perform detailed cluster analysis
    cluster_metadata = analyze_clusters_in_detail(
        all_explanations, clusters, n_clusters, vectorizer, 
        explanation_sources, valid_indices, data
    )
    
    print("\nAnalysis complete! Results saved to the ClusterAnalysis directory.")
