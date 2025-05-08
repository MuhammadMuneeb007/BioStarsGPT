import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pandas as pd
from collections import Counter
import os
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(filename):
    """Load and parse the JSON data file."""
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            # Try loading as JSONL
            file.seek(0)
            data = [json.loads(line) for line in file if line.strip()]
    
    return data

def create_prompt_completion_pairs(data):
    """Convert Question1/2 and Explanation1/2 to prompt/completion pairs with group information."""
    processed_data = []
    
    for i, item in enumerate(data):
        # Create a unique group ID for each question pair
        group_id = item.get('QuestionCount', i)
        
        # Process Question1/Explanation1 pair
        if 'Question1' in item and 'Explanation1' in item:
            processed_data.append({
                'prompt': item['Question1'],
                'completion': item['Explanation1'],
                'group_id': group_id,
                'pair_type': 'question1',
                'complexity': item.get('complexity', 'Unknown')
            })
        
        # Process Question2/Explanation2 pair
        if 'Question2' in item and 'Explanation2' in item:
            processed_data.append({
                'prompt': item['Question2'],
                'completion': item['Explanation2'],
                'group_id': group_id,
                'pair_type': 'question2',
                'complexity': item.get('complexity', 'Unknown')
            })
    
    return processed_data

def create_semantic_embeddings(data):
    """Create semantic embeddings using a pre-trained language model."""
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract prompts
    prompts = [item['prompt'] for item in data]
    
    # Generate embeddings
    print("Generating semantic embeddings...")
    embeddings = model.encode(prompts, show_progress_bar=True)
    
    return embeddings

def semantic_clustering(embeddings, min_cluster_size=None):
    """Perform semantic clustering using UMAP dimensionality reduction and HDBSCAN with optimal cluster parameters."""
    # First reduce dimensionality with UMAP
    print("Performing dimensionality reduction with UMAP...")
    umap_embeddings = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric='cosine',
        random_state=42
    ).fit_transform(embeddings)
    
    # Find optimal min_cluster_size if not provided
    if min_cluster_size is None:
        print("Finding optimal number of clusters...")
        best_score = -1
        best_min_cluster_size = 0
        best_labels = None
        
        # Try a range of min_cluster_size values
        min_sizes_to_try = range(5, min(100, len(embeddings) // 20), 5)
        for min_size in min_sizes_to_try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            clusterer.fit(umap_embeddings)
            
            # Get the number of clusters and non-noise points
            unique_clusters = set(clusterer.labels_)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            non_noise_mask = clusterer.labels_ != -1
            
            # Only evaluate clustering if we have more than one cluster and some non-noise points
            if n_clusters > 1 and np.sum(non_noise_mask) > min_size:
                try:
                    # Use silhouette score instead of DBCV
                    non_noise_data = umap_embeddings[non_noise_mask]
                    non_noise_labels = clusterer.labels_[non_noise_mask]
                    score = silhouette_score(non_noise_data, non_noise_labels)
                    
                    print(f"Min cluster size {min_size} produced {n_clusters} clusters with silhouette score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_min_cluster_size = min_size
                        best_labels = clusterer.labels_
                except Exception as e:
                    print(f"Error calculating score for min_size={min_size}: {e}")
        
        if best_min_cluster_size > 0:
            print(f"Optimal min_cluster_size: {best_min_cluster_size} with silhouette score: {best_score:.4f}")
            labels = best_labels
        else:
            # Fallback to default if optimization fails
            print("Optimization failed, using default min_cluster_size=10")
            min_cluster_size = 10
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            labels = clusterer.labels_
    else:
        # Use provided min_cluster_size
        print(f"Using provided min_cluster_size={min_cluster_size}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(umap_embeddings)
        labels = clusterer.labels_
    
    # Get cluster labels (including outliers labeled as -1)
    outlier_count = sum(labels == -1)
    print(f"Found {outlier_count} outliers ({outlier_count/len(labels)*100:.1f}% of data)")
    
    # Count non-outlier clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Created {num_clusters} semantic clusters (excluding outliers)")
    
    return labels

def split_data_by_clusters_and_groups(data, clusters, train_size=0.8, val_size=10000, test_size=100, remove_outliers=False):
    """
    Split data ensuring proportional representation from each cluster and keeping question pairs together.
    
    Parameters:
    -----------
    data : list
        The data to split
    clusters : numpy.ndarray
        The cluster assignments including outliers (-1)
    train_size : float
        The proportion of data to allocate to the training set
    val_size : int
        The size of the validation set
    test_size : int
        The size of the test set
    remove_outliers : bool
        Whether to remove outliers from the datasets
    """
    # If remove_outliers is True, filter out outliers
    if remove_outliers:
        print("Removing outliers from all datasets...")
        # Create a mask for non-outlier data
        non_outlier_mask = clusters != -1
        # Filter data and clusters
        filtered_indices = np.where(non_outlier_mask)[0]
        filtered_data = [data[i] for i in filtered_indices]
        filtered_clusters = clusters[non_outlier_mask]
        
        # Continue with filtered data
        data_to_process = filtered_data
        clusters_to_process = filtered_clusters
        
        # Count removed outliers
        removed_count = len(data) - len(filtered_data)
        print(f"Removed {removed_count} outliers ({removed_count/len(data)*100:.1f}% of data)")
    else:
        print("Keeping outliers in all datasets...")
        # Use all data
        data_to_process = data
        clusters_to_process = clusters
    
    # Group data by group_id
    groups = {}
    for i, item in enumerate(data_to_process):
        group_id = item['group_id']
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)
    
    # Calculate average cluster for each group
    group_clusters = {}
    for group_id, indices in groups.items():
        group_clusters[group_id] = int(np.mean([clusters_to_process[idx] for idx in indices]))
    
    # Create a dictionary of cluster to groups
    cluster_groups = {}
    for group_id, cluster in group_clusters.items():
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(group_id)
    
    # Calculate how many groups to take from each cluster
    total_groups = len(groups)
    test_groups = []
    val_groups = []
    train_groups = []
    
    # Approximate sizes based on groups not individual samples
    approx_test_groups = max(1, int(test_size / 2))  # Assuming ~2 samples per group
    approx_val_groups = max(1, int(val_size / 2))    # Assuming ~2 samples per group
    
    # First, allocate test set (smallest)
    for cluster, group_ids in cluster_groups.items():
        # Calculate proportional allocation for test set
        n_test_groups = max(1, int(len(group_ids) * (approx_test_groups / total_groups)))
        if len(test_groups) + n_test_groups > approx_test_groups:
            n_test_groups = approx_test_groups - len(test_groups)
        
        if n_test_groups <= 0:
            continue
            
        selected_groups = random.sample(group_ids, min(n_test_groups, len(group_ids)))
        test_groups.extend(selected_groups)
        
        # Remove selected groups from the cluster
        cluster_groups[cluster] = [gid for gid in group_ids if gid not in selected_groups]
        
        if len(test_groups) >= approx_test_groups:
            break
    
    # Next, allocate validation set
    remaining_clusters = [c for c in cluster_groups.keys() if cluster_groups[c]]
    for cluster in remaining_clusters:
        group_ids = cluster_groups[cluster]
        # Calculate proportional allocation for validation set
        n_val_groups = max(1, int(len(group_ids) * (approx_val_groups / (total_groups - len(test_groups)))))
        if len(val_groups) + n_val_groups > approx_val_groups:
            n_val_groups = approx_val_groups - len(val_groups)
        
        if n_val_groups <= 0:
            continue
            
        selected_groups = random.sample(group_ids, min(n_val_groups, len(group_ids)))
        val_groups.extend(selected_groups)
        
        # Remove selected groups from the cluster
        cluster_groups[cluster] = [gid for gid in group_ids if gid not in selected_groups]
        
        if len(val_groups) >= approx_val_groups:
            break
    
    # Allocate remaining groups to train set
    for cluster in cluster_groups:
        train_groups.extend(cluster_groups[cluster])
    
    # Convert groups to indices
    test_indices = [idx for group in test_groups for idx in groups[group]]
    val_indices = [idx for group in val_groups for idx in groups[group]]
    train_indices = [idx for group in train_groups for idx in groups[group]]
    
    # Ensure we have the right number of samples in each set
    # If we need more samples, we can add more groups
    if len(test_indices) < test_size and len(train_groups) > 0:
        additional_groups = []
        while len(test_indices) < test_size and len(train_groups) > 0:
            group = train_groups.pop(0)
            additional_groups.append(group)
            test_indices.extend(groups[group])
        
        # Remove these groups from train set
        for group in additional_groups:
            train_indices = [idx for idx in train_indices if idx not in groups[group]]
    
    if len(val_indices) < val_size and len(train_groups) > 0:
        additional_groups = []
        while len(val_indices) < val_size and len(train_groups) > 0:
            group = train_groups.pop(0)
            additional_groups.append(group)
            val_indices.extend(groups[group])
        
        # Remove these groups from train set
        for group in additional_groups:
            train_indices = [idx for idx in train_indices if idx not in groups[group]]
    
    # Create the datasets
    train_data = [data_to_process[i] for i in train_indices]
    val_data = [data_to_process[i] for i in val_indices]
    test_data = [data_to_process[i] for i in test_indices]
    
    # Create mapping from filtered indices to original indices if outliers were removed
    if remove_outliers:
        # Map each split back to the original data indices
        train_original_indices = [filtered_indices[i] for i in train_indices]
        val_original_indices = [filtered_indices[i] for i in val_indices]
        test_original_indices = [filtered_indices[i] for i in test_indices]
        
        # Return original indices for visualization purposes
        return train_data, val_data, test_data, train_original_indices, val_original_indices, test_original_indices
    else:
        # If outliers were kept, indices are already correct
        return train_data, val_data, test_data, train_indices, val_indices, test_indices

def analyze_semantic_distribution(train_data, val_data, test_data, embeddings, clusters, train_indices, val_indices, test_indices, remove_outliers=False):
    """
    Create an enhanced publication-quality visualization of semantic distribution across datasets.
    
    Parameters:
    -----------
    train_data, val_data, test_data : lists
        The split datasets containing the items
    embeddings : numpy.ndarray
        The embeddings of all items
    clusters : numpy.ndarray
        The cluster assignments including outliers (-1)
    train_indices, val_indices, test_indices : lists
        The indices of the data in each split, referring to positions in the original data
    remove_outliers : bool
        Whether outliers were removed from the datasets
    """
 
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 600
    })
    
    # Get indices for each set
    all_data = train_data + val_data + test_data
    
    # Create 2D projection for visualization with optimized parameters
    print("Creating UMAP projection for visualization...")
    umap_2d = umap.UMAP(
        n_neighbors=20,
        n_components=2,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    ).fit_transform(embeddings)
    
    # Enhanced color scheme suitable for publication (colorblind-friendly with improved contrast)
    train_color = '#0173B2'  # Blue
    val_color = '#029E73'    # Green
    test_color = '#D55E00'   # Orange-red
    outlier_color = '#CC78BC'  # Purple for outliers
    
    # Create a figure with higher resolution
    fig = plt.figure(figsize=(12, 8))
    
    # Create gridspec for plot and legend with better proportions
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Main plot area with improved styling
    ax = fig.add_subplot(gs[0])
    
    # Add enhanced background and styling
    ax.set_facecolor('#F8F8F8')
    ax.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')
    
    # Identify clusters for visualization
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Exclude outlier cluster
    
    # Map clusters to colors for better visualization (excluding -1 for outliers)
    cmap = plt.cm.viridis
    cluster_colors = {c: cmap(i / max(1, num_clusters)) for i, c in enumerate(unique_clusters) if c != -1}
    
    # Identify outliers and non-outliers in each dataset
    # When remove_outliers=True, these lists should be empty since outliers were removed
    train_outlier_indices = [i for i in train_indices if clusters[i] == -1]
    val_outlier_indices = [i for i in val_indices if clusters[i] == -1]
    test_outlier_indices = [i for i in test_indices if clusters[i] == -1]
    
    train_non_outlier_indices = [i for i in train_indices if clusters[i] != -1]
    val_non_outlier_indices = [i for i in val_indices if clusters[i] != -1]
    test_non_outlier_indices = [i for i in test_indices if clusters[i] != -1]
    
    # Add enhanced density contours with better transparency and granularity
    def add_density_contours(points, color, alpha=0.2, levels=7):
        if len(points) < 10:
            return
            
        x, y = points[:, 0], points[:, 1]
        
        # Calculate the point density with improved bandwidth
        xy = np.vstack([x, y])
        try:
            # Use Scott's rule for bandwidth selection for better contour definition
            kernel = gaussian_kde(xy, bw_method='scott')
            
            # Create a meshgrid with more points for smoother contours
            margin = 0.1
            x_min, x_max = min(x) - margin, max(x) + margin
            y_min, y_max = min(y) - margin, max(y) + margin
            xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Get the density values
            z = np.reshape(kernel(positions).T, xx.shape)
            
            # Create custom colormap for better visualization
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [color, color], N=100)
            
            # Plot contours with improved styling
            contour = ax.contour(xx, yy, z, levels=levels, colors=color, 
                               alpha=alpha*1.5, linewidths=0.8)
            contourf = ax.contourf(xx, yy, z, levels=levels, cmap=custom_cmap, 
                                 alpha=alpha)
        except Exception as e:
            print(f"Could not create contours: {e}")
    
    # Visualize cluster distribution with improved styling
    print("Plotting improved data visualization...")
    
    # Create scatter plot handles for legend
    scatter_handles = []
    
    # Add cluster labels as a semi-transparent background
    if num_clusters > 1:  # Only if we have more than one non-outlier cluster
        # Plot points colored by cluster for the background context
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip outliers for this background layer
                continue
                
            # Get indices for this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                ax.scatter(
                    umap_2d[cluster_indices, 0], umap_2d[cluster_indices, 1],
                    c=[cluster_colors[cluster_id]], s=15, alpha=0.1,
                    edgecolors='none', zorder=1
                )
    
    # Add density contours for each dataset (non-outliers only)
    if len(train_non_outlier_indices) > 10:
        add_density_contours(umap_2d[train_non_outlier_indices], train_color, alpha=0.15, levels=8)
    if len(val_non_outlier_indices) > 10:
        add_density_contours(umap_2d[val_non_outlier_indices], val_color, alpha=0.15, levels=8)
    if len(test_non_outlier_indices) > 10:
        add_density_contours(umap_2d[test_non_outlier_indices], test_color, alpha=0.15, levels=8)
    
    # Plot the non-outlier scatter points with improved styling
    if train_non_outlier_indices:
        train_scatter = ax.scatter(
            umap_2d[train_non_outlier_indices, 0], umap_2d[train_non_outlier_indices, 1],
            c=train_color, s=40, alpha=0.7, edgecolors='white', linewidths=0.3, zorder=10
        )
        scatter_handles.append((train_scatter, f'Training (n={len(train_indices)})'))
    
    if val_non_outlier_indices:
        val_scatter = ax.scatter(
            umap_2d[val_non_outlier_indices, 0], umap_2d[val_non_outlier_indices, 1],
            c=val_color, s=50, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=11
        )
        scatter_handles.append((val_scatter, f'Validation (n={len(val_indices)})'))
    
    if test_non_outlier_indices:
        test_scatter = ax.scatter(
            umap_2d[test_non_outlier_indices, 0], umap_2d[test_non_outlier_indices, 1],
            c=test_color, s=60, alpha=0.9, edgecolors='white', linewidths=0.3, zorder=12
        )
        scatter_handles.append((test_scatter, f'Test (n={len(test_indices)})'))
    
    # Plot outliers with star markers and enhanced styling ONLY if outliers are kept
    # This is the fixed part - only show outliers in the plot if remove_outliers=False
    if  remove_outliers:
        if train_outlier_indices:
            train_outlier_scatter = ax.scatter(
                umap_2d[train_outlier_indices, 0], umap_2d[train_outlier_indices, 1],
                c=outlier_color, s=70, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=20
            )
            scatter_handles.append((train_outlier_scatter, f'Training Outliers (n={len(train_outlier_indices)})'))
        
        if val_outlier_indices:
            val_outlier_scatter = ax.scatter(
                umap_2d[val_outlier_indices, 0], umap_2d[val_outlier_indices, 1],
                c=outlier_color, s=80, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=21
            )
            scatter_handles.append((val_outlier_scatter, f'Validation Outliers (n={len(val_outlier_indices)})'))
        
        if test_outlier_indices:
            test_outlier_scatter = ax.scatter(
                umap_2d[test_outlier_indices, 0], umap_2d[test_outlier_indices, 1],
                c=outlier_color, s=90, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=22
            )
            scatter_handles.append((test_outlier_scatter, f'Test Outliers (n={len(test_outlier_indices)})'))
    
    # Set axis limits with better padding
    x_min, x_max = np.min(umap_2d[:, 0])-1.5, np.max(umap_2d[:, 0])+1.5
    y_min, y_max = np.min(umap_2d[:, 1])-1.5, np.max(umap_2d[:, 1])+1.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add statistics about total clusters
    num_clusters_text = f"Total clusters: {num_clusters}"
    ax.text(0.02, 0.98, num_clusters_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Calculate diversity metrics
    def calculate_diversity(indices):
        if len(indices) < 2:
            return 0
        
        subset_embeddings = embeddings[indices]
        similarities = cosine_similarity(subset_embeddings)
        np.fill_diagonal(similarities, 0)
        return 1 - similarities.sum() / (len(indices) * (len(indices) - 1))
    
    train_diversity = calculate_diversity(train_indices)
    val_diversity = calculate_diversity(val_indices)
    test_diversity = calculate_diversity(test_indices)
    
    # Calculate outlier percentage in each set
    train_outlier_pct = len(train_outlier_indices) / len(train_indices) * 100 if train_indices else 0
    val_outlier_pct = len(val_outlier_indices) / len(val_indices) * 100 if val_indices else 0
    test_outlier_pct = len(test_outlier_indices) / len(test_indices) * 100 if test_indices else 0
    
    # Formatting and labels with improved typography
    ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
    outlier_status = "with Outliers Removed" if remove_outliers else "with Outliers Included"
    ax.set_title(f'Semantic Distribution of Question Datasets ({outlier_status})', pad=15, fontweight='bold')
    
    # Remove tick labels to reduce clutter but improve tick visibility
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    
    # Fine-tune aesthetics for spines with enhanced styling
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.2)
    
    # Create legend axis (outside the main plot) with improved styling
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    
    # Create custom legend with better spacing and organization
    handles = [h[0] for h in scatter_handles]
    labels = [h[1] for h in scatter_handles]
    
    legend = legend_ax.legend(
        handles, 
        labels,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        title='Dataset Distribution',
        borderpad=1.0,
        labelspacing=1.2
    )
    legend.get_title().set_fontweight('bold')
    
    # Add information panel with improved formatting showing the diversity metrics and outlier percentages
    outlier_status_text = "Outlier Percentages (Removed from Dataset):" if remove_outliers else "Outlier Percentages (Included in Dataset):"
    
    info_text = (
        f"Semantic Diversity Metrics:\n"
        f"Training set: {train_diversity:.4f}\n"
        f"Validation set: {val_diversity:.4f}\n"
        f"Test set: {test_diversity:.4f}\n\n"
        f"{outlier_status_text}\n"
        f"Training set: {train_outlier_pct:.1f}%\n"
        f"Validation set: {val_outlier_pct:.1f}%\n"
        f"Test set: {test_outlier_pct:.1f}%"
    )
    
    # Add the info box with improved styling
    info_box = legend_ax.text(
        0.05, 0.5, info_text, 
        transform=legend_ax.transAxes,
        fontsize=10, 
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', edgecolor='#CCCCCC', alpha=0.95)
    )
    
    # Add explanation about outliers
    if remove_outliers:
        outlier_note = (
            f"Note: Outliers were removed from all datasets.\n"
            f"They represented semantically unique questions\n"
            f"that didn't fit well into any cluster."
        )
    else:
        outlier_note = (
            f"Note: Outliers are included in all datasets.\n"
            f"They represent semantically unique questions\n"
            f"that don't fit well into any cluster."
        )
    
    legend_ax.text(
        0.05, 0.10, outlier_note, 
        transform=legend_ax.transAxes,
        fontsize=9, 
        style='italic',
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    
    # Save in multiple formats for publication with increased quality
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    outlier_filename = "without_outliers" if remove_outliers else "with_outliers"
    print(f"Saving enhanced visualization in multiple formats...")
    plt.savefig(f'{output_dir}/semantic_distribution_{outlier_filename}.png', dpi=600, bbox_inches='tight')
    
    outlier_status_message = "with outliers removed" if remove_outliers else "with outliers included"
    print(f"Enhanced publication-quality visualization saved {outlier_status_message}")
    
    # Print diversity results
    print("\nSemantic Diversity Results:")
    print(f"Train set: {train_diversity:.4f}")
    print(f"Validation set: {val_diversity:.4f}")
    print(f"Test set: {test_diversity:.4f}")
    
    # Print outlier results
    outlier_status_print = "removed from datasets" if remove_outliers else "included in datasets"
    print(f"\nOutlier Percentages ({outlier_status_print}):")
    print(f"Train set: {train_outlier_pct:.1f}% ({len(train_outlier_indices)} out of {len(train_indices)})")
    print(f"Validation set: {val_outlier_pct:.1f}% ({len(val_outlier_indices)} out of {len(val_indices)})")
    print(f"Test set: {test_outlier_pct:.1f}% ({len(test_outlier_indices)} out of {len(test_indices)})")
    print(f"Total clusters: {num_clusters}")
    
    return {
        'train_diversity': train_diversity,
        'val_diversity': val_diversity,
        'test_diversity': test_diversity,
        'train_outlier_pct': train_outlier_pct,
        'val_outlier_pct': val_outlier_pct,
        'test_outlier_pct': test_outlier_pct,
        'num_clusters': num_clusters,
        'figure': fig  # Return the figure for further adjustments if needed
    }

def verify_question_pairs(train_data, val_data, test_data, remove_outliers=False):
    """Verify that question pairs are kept together and create visualization."""
    # Group data by group_id in each dataset
    train_groups = {}
    val_groups = {}
    test_groups = {}
    
    for item in train_data:
        if item['group_id'] not in train_groups:
            train_groups[item['group_id']] = []
        train_groups[item['group_id']].append(item['pair_type'])
    
    for item in val_data:
        if item['group_id'] not in val_groups:
            val_groups[item['group_id']] = []
        val_groups[item['group_id']].append(item['pair_type'])
    
    for item in test_data:
        if item['group_id'] not in test_groups:
            test_groups[item['group_id']] = []
        test_groups[item['group_id']].append(item['pair_type'])
    
    # Check for any overlap
    all_groups = set(list(train_groups.keys()) + list(val_groups.keys()) + list(test_groups.keys()))
    split_issues = 0
    
    for group in all_groups:
        datasets_with_group = 0
        if group in train_groups:
            datasets_with_group += 1
        if group in val_groups:
            datasets_with_group += 1
        if group in test_groups:
            datasets_with_group += 1
        
        if datasets_with_group > 1:
            split_issues += 1
    
    if split_issues == 0:
        print("✓ All question pairs are kept together in the same dataset!")
    else:
        print(f"✗ Found {split_issues} groups split across multiple datasets.")
    
    # Count complete pairs (both question1 and question2) in each dataset
    train_complete = sum(1 for types in train_groups.values() if 'question1' in types and 'question2' in types)
    val_complete = sum(1 for types in val_groups.values() if 'question1' in types and 'question2' in types)
    test_complete = sum(1 for types in test_groups.values() if 'question1' in types and 'question2' in types)
    
    print(f"\nComplete question pairs (both Q1 and Q2):")
    print(f"Train set: {train_complete} complete pairs")
    print(f"Validation set: {val_complete} complete pairs")
    print(f"Test set: {test_complete} complete pairs")
    
    # Count single questions in each dataset
    train_q1_only = sum(1 for types in train_groups.values() if 'question1' in types and 'question2' not in types)
    train_q2_only = sum(1 for types in train_groups.values() if 'question2' in types and 'question1' not in types)
    val_q1_only = sum(1 for types in val_groups.values() if 'question1' in types and 'question2' not in types)
    val_q2_only = sum(1 for types in val_groups.values() if 'question2' in types and 'question1' not in types)
    test_q1_only = sum(1 for types in test_groups.values() if 'question1' in types and 'question2' not in types)
    test_q2_only = sum(1 for types in test_groups.values() if 'question2' in types and 'question1' not in types)
    
    print(f"\nIncomplete question pairs:")
    print(f"Train set: {train_q1_only} Q1-only, {train_q2_only} Q2-only")
    print(f"Validation set: {val_q1_only} Q1-only, {val_q2_only} Q2-only")
    print(f"Test set: {test_q1_only} Q1-only, {test_q2_only} Q2-only")
    
    # Create a publication-quality chart displaying the distribution of pairs
    plt.figure(figsize=(10, 7))
    
    # Use GridSpec for external legend
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Main axis for the bar chart
    ax = plt.subplot(gs[0])
    
    # Plot the bars with handles for legend
    labels = ['Train', 'Validation', 'Test']
    complete_pairs = [train_complete, val_complete, test_complete]
    q1_only = [train_q1_only, val_q1_only, test_q1_only]
    q2_only = [train_q2_only, val_q2_only, test_q2_only]
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, complete_pairs, width, color='#3274A1', alpha=0.8)
    bars2 = ax.bar(x, q1_only, width, color='#E1812C', alpha=0.8)
    bars3 = ax.bar(x + width, q2_only, width, color='#3A923A', alpha=0.8)
    
    # Calculate totals for percentage calculations
    totals = [
        train_complete + train_q1_only + train_q2_only,
        val_complete + val_q1_only + val_q2_only,
        test_complete + test_q1_only + test_q2_only
    ]
    
    # Add percentage annotations
    def add_percentage(bars, totals):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / totals[i] * 100) if totals[i] > 0 else 0
            ax.annotate(f'{height}\n({percentage:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_percentage(bars1, totals)
    add_percentage(bars2, totals)
    add_percentage(bars3, totals)
    
    # Customize the chart
    outlier_status = "without Outliers" if remove_outliers else "with Outliers"
    ax.set_title(f'Question Pair Distribution Across Datasets ({outlier_status})', fontsize=16, weight='bold')
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend axis
    legend_ax = plt.subplot(gs[1])
    legend_ax.axis('off')
    
    # Add legend with descriptive text
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#3274A1', alpha=0.8, label='Complete pairs (Q1+Q2)'),
        plt.Rectangle((0, 0), 1, 1, color='#E1812C', alpha=0.8, label='Q1 only'),
        plt.Rectangle((0, 0), 1, 1, color='#3A923A', alpha=0.8, label='Q2 only')
    ]
    
    legend = legend_ax.legend(
        handles=legend_elements, 
        loc='upper left',
        title='Pair Types',
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC'
    )
    legend.get_title().set_fontweight('bold')
    
    # Add explanatory text
    outlier_note = "outliers removed" if remove_outliers else "outliers included"
    explanation_text = (
        f"Question Pair Analysis ({outlier_note}):\n\n"
        f"• Complete pairs: questions with both\n  a primary (Q1) and follow-up (Q2)\n  version\n\n"
        f"• Q1 only: questions with only\n  a primary version\n\n"
        f"• Q2 only: questions with only\n  a follow-up version\n\n"
        f"Note: All question pairs are kept\n"
        f"together in the same dataset split."
    )
    
    legend_ax.text(0.05, 0.5, explanation_text, transform=legend_ax.transAxes,
                 fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    outlier_filename = "without_outliers" if remove_outliers else "with_outliers"
    plt.savefig(f'{output_dir}/pair_distribution_{outlier_filename}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pair_distribution_{outlier_filename}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/pair_distribution_{outlier_filename}.svg', format='svg', bbox_inches='tight')
    
    outlier_message = "without outliers" if remove_outliers else "with outliers"
    print(f"Question pair distribution visualization saved {outlier_message}")

def save_to_jsonl(data, filename):
    """Save data to a JSONL file."""
    # Create output directory if it doesn't exist
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Create a copy of the data without the internal fields used for grouping
    clean_data = []
    for item in data:
        clean_item = item.copy()
        if 'group_id' in clean_item:
            del clean_item['group_id']
        if 'pair_type' in clean_item:
            del clean_item['pair_type']
        clean_data.append(clean_item)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in clean_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(clean_data)} items to {output_path}")

def main(remove_outliers=False):
    input_file = "filtered_questions_intermediate.json"
    
    # Create output directory
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and parse the datafseaborn-whitegrid
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    # For debugging with smaller dataset, uncomment the next line
    #data = data[:10000]
    print(f"Loaded {len(data)} items")
    
    # Convert to prompt/completion pairs with group information
    processed_data = create_prompt_completion_pairs(data)
    print(f"Created {len(processed_data)} prompt/completion pairs")
    
    # Create semantic embeddings
    embeddings = create_semantic_embeddings(processed_data)
    
    # Perform semantic clustering with automatic cluster determination
    print("Performing semantic clustering with automatic cluster determination...")
    clusters = semantic_clustering(embeddings)  # No min_cluster_size - will find optimal automatically
    
    # Calculate actual sizes for train/val/test
    total_samples = len(processed_data)
    # Ensure validation and test sets don't exceed available data
    val_size = max(20000, total_samples * 0.15)  # 10000 questions or 15%, whichever is smaller
    test_size = min(100, total_samples * 0.01)    # 100 questions or 1%, whichever is smaller
    
    outlier_status = "removing outliers" if remove_outliers else "keeping outliers"
    print(f"Splitting data into train, validation, and test sets ({outlier_status})...")
    print(f"Target sizes: train (~{total_samples - val_size - test_size:.0f} samples), "
          f"validation (~{val_size:.0f} samples), and test (~{test_size:.0f} samples)")
    
    # Split the data using the function with outlier removal option
    train_data, val_data, test_data, train_indices, val_indices, test_indices = split_data_by_clusters_and_groups(
        processed_data, clusters, 
        train_size=0.8, 
        val_size=int(val_size), 
        test_size=int(test_size),
        remove_outliers=remove_outliers
    )
    
    print(f"Final split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    
    # Verify that question pairs stayed together
    verify_question_pairs(train_data, val_data, test_data, remove_outliers=remove_outliers)
    
    # Analyze semantic distribution with improved visualization
    print("Analyzing semantic distribution across splits...")
    
    # Use full original embeddings for visualization (handle differently based on outlier removal)
    distribution_results = analyze_semantic_distribution(
        train_data, val_data, test_data, 
        embeddings, clusters, 
        train_indices, val_indices, test_indices,
        remove_outliers=remove_outliers
    )
    
    # Create filenames with outlier status
    outlier_suffix = "_without_outliers" if remove_outliers else "_with_outliers"
    
    # Save to JSONL files
    save_to_jsonl(train_data, f'Train{outlier_suffix}.jsonl')
    save_to_jsonl(val_data, f'Validation{outlier_suffix}.jsonl')
    save_to_jsonl(test_data, f'Test{outlier_suffix}.jsonl')
    
    # Print outlier statistics
    outlier_status_print = "removed from" if remove_outliers else "included in"
    
    # Identify outliers in each dataset
    train_outlier_indices = [i for i in train_indices if clusters[i] == -1]
    val_outlier_indices = [i for i in val_indices if clusters[i] == -1]
    test_outlier_indices = [i for i in test_indices if clusters[i] == -1]
    
    print(f"\nData processing complete!")
    print(f"Train set: {len(train_data)} questions ({len(train_data)/len(processed_data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} questions ({len(val_data)/len(processed_data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} questions ({len(test_data)/len(processed_data)*100:.1f}%)")
    
    print(f"\nOutlier statistics ({outlier_status_print} datasets):")
    print(f"Train set: {len(train_outlier_indices)} outliers ({len(train_outlier_indices)/len(train_indices)*100 if train_indices else 0:.1f}%)")
    print(f"Validation set: {len(val_outlier_indices)} outliers ({len(val_outlier_indices)/len(val_indices)*100 if val_indices else 0:.1f}%)")
    print(f"Test set: {len(test_outlier_indices)} outliers ({len(test_outlier_indices)/len(test_indices)*100 if test_indices else 0:.1f}%)")
    print(f"Total clusters found: {distribution_results['num_clusters']}")

if __name__ == "__main__":
    # Set remove_outliers to False to keep outliers (default behavior)
    # Set to True to remove outliers from all datasets
    remove_outliers = False
    main(remove_outliers=remove_outliers)