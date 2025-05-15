import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from utils.dataset import get_classnames_from_folder

def load_features(file_path):
    """Load the saved features from .pt file"""
    return torch.load(file_path)

def prepare_features_for_tsne(features, block_idx, class_name, domains):
    """
    Extract features for a specific block and class across all domains
    Returns a numpy array of shape (n_samples, n_features) and domain labels
    
    Args:
        features: Dictionary with structure {block_idx: {domain: {class_name: [tensor1, tensor2, ...]}}}
        block_idx: Index of the transformer block
        class_name: Name of the class to analyze (e.g., "Alarm_Clock")
        domains: List of domains to include
    """
    all_features = []
    domain_labels = []
    
    # Collect features from all domains for the specified class
    for domain in domains:
        if domain in features[block_idx] and class_name in features[block_idx][domain]:
            domain_features = features[block_idx][domain][class_name]
            # Convert list of tensors to numpy array, handling [1, 768] shape
            features_array = np.stack([f.numpy().reshape(-1) for f in domain_features])
            all_features.append(features_array)
            domain_labels.extend([domain] * len(domain_features))
    
    # Combine features from all domains
    if all_features:
        features_array = np.vstack(all_features)
    else:
        features_array = np.array([])
    
    return features_array, domain_labels

def plot_tsne_multi_blocks(features, block_indices, class_name, domains, save_path=None):
    n_blocks = len(block_indices)
    ncols = min(n_blocks, 3)
    nrows = (n_blocks + ncols - 1) // ncols

    plt.figure(figsize=(6 * ncols, 5 * nrows))
    unique_domains = sorted(set(domains))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
    domain_to_color = {dom: color for dom, color in zip(unique_domains, colors)}

    for i, block_idx in enumerate(block_indices):
        features_array, domain_labels = prepare_features_for_tsne(features, block_idx, class_name, domains)
        if features_array is None or len(features_array) == 0:
            continue
        
        features_scaled = StandardScaler().fit_transform(features_array)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(features_scaled)
        
        ax = plt.subplot(nrows, ncols, i + 1)
        for dom in unique_domains:
            idxs = [j for j, d in enumerate(domain_labels) if d == dom]
            if idxs:  # Only plot if we have points for this domain
                ax.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], label=dom, color=domain_to_color[dom], alpha=0.7)
        ax.set_title(f"Block {block_idx}")
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if i == 0:
            ax.legend()
    
    plt.suptitle(f"t-SNE of CLS Tokens for Class {class_name} (All Domains)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    features_file = "clip_block_features_20250515_092209.pt"
    data_dir = "data/officehome"  # Path to your dataset
    block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    domains = ['art', 'clipart', 'product', 'real_world']
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    features = load_features(features_file)
    
    # Get class names from the dataset
    classnames = get_classnames_from_folder(data_dir, domains[0])
    print("\nAvailable classes:")
    for idx, name in enumerate(classnames):
        print(f"{idx}: {name}")
    
    # Use the first class name as an example
    class_name = classnames[0]  # This will be the first class name from the dataset
    print(f"\nUsing class: {class_name}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(vis_dir, f"tsne_multiblock_{class_name}_{timestamp}.png")
    plot_tsne_multi_blocks(features, block_indices, class_name, domains, save_path)
    print(f"\nt-SNE multi-block plot saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main() 