import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import random
import clip
from utils.dataset import get_classnames_from_folder

def load_features(file_path):
    return torch.load(file_path)

def prepare_features_for_tsne_by_class(features, block_idx, domains, max_total_samples=1000):
    all_samples = []
    for domain in domains:
        if domain in features[block_idx]:
            for cls in features[block_idx][domain]:
                for f in features[block_idx][domain][cls]:
                    all_samples.append((f.numpy().reshape(-1), cls, domain))
    random.shuffle(all_samples)
    if len(all_samples) > max_total_samples:
        all_samples = all_samples[:max_total_samples]
    features_array = np.stack([s[0] for s in all_samples])
    class_labels = [s[1] for s in all_samples]
    domain_labels = [s[2] for s in all_samples]
    class_names = sorted(set(class_labels))
    return features_array, class_labels, domain_labels

def plot_tsne_multi_blocks_by_class(features, block_indices, domains, classnames, max_total_samples=1000, save_path=None, color_by_domain=False):
    n_blocks = len(block_indices)
    ncols = min(n_blocks, 3)
    nrows = (n_blocks + ncols - 1) // ncols

    plt.figure(figsize=(6 * ncols, 5 * nrows))
    
    if color_by_domain:
        # Color by domain
        colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        color_map = {dom: color for dom, color in zip(domains, colors)}
        title_suffix = "by Domain"
    else:
        # Color by class
        colors = plt.cm.tab20(np.linspace(0, 1, len(classnames)))
        color_map = {name: color for name, color in zip(classnames, colors)}
        title_suffix = "by Class"

    for i, block_idx in enumerate(block_indices):
        features_array, class_labels, domain_labels = prepare_features_for_tsne_by_class(features, block_idx, domains, max_total_samples)
        if features_array is None or len(features_array) == 0:
            continue
        
        features_scaled = StandardScaler().fit_transform(features_array)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(features_scaled)
        
        ax = plt.subplot(nrows, ncols, i + 1)
        if color_by_domain:
            # Plot points colored by domain
            for dom in domains:
                idxs = [j for j, d in enumerate(domain_labels) if d == dom]
                if len(idxs) > 0:
                    ax.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], 
                             color=color_map[dom], alpha=0.7, s=10, label=dom)
        else:
            # Plot points colored by class
            for cls in classnames:
                idxs = [j for j, c in enumerate(class_labels) if c == cls]
                if len(idxs) > 0:
                    ax.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], 
                             color=color_map[cls], alpha=0.7, s=10, label=cls)
        
        ax.set_title(f"Block {block_idx}")
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if i == 0:  # Only show legend in first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f"t-SNE of CLS Tokens {title_suffix}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    features_file = "clip_block_features_20250515_092209.pt"
    data_dir = "data/officehome"
    domain_for_classnames = "art"  # Using art domain to get class names
    block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    domains = ['art', 'clipart', 'product', 'real_world']
    max_total_samples = 1000
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    features = load_features(features_file)
    classnames = get_classnames_from_folder(data_dir, domain_for_classnames)
    
    # Create plots with both colorings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot colored by class
    save_path_class = os.path.join(vis_dir, f"tsne_multiblock_byclass_{timestamp}.png")
    plot_tsne_multi_blocks_by_class(features, block_indices, domains, classnames, 
                                   max_total_samples, save_path_class, color_by_domain=False)
    print(f"\nt-SNE multi-block plot (colored by class) saved to: {os.path.abspath(save_path_class)}")
    
    # Plot colored by domain
    save_path_domain = os.path.join(vis_dir, f"tsne_multiblock_bydomain_{timestamp}.png")
    plot_tsne_multi_blocks_by_class(features, block_indices, domains, classnames, 
                                   max_total_samples, save_path_domain, color_by_domain=True)
    print(f"t-SNE multi-block plot (colored by domain) saved to: {os.path.abspath(save_path_domain)}")

if __name__ == "__main__":
    main() 