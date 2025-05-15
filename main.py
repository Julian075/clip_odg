import os
import torch
import argparse
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import clip

from src.analysis.analyze_clustering import plot_tsne_multi_blocks
from src.analysis.analyze_tsne_by_class import plot_tsne_multi_blocks_by_class
from utils.dataset import get_classnames_from_folder, load_datasets
from utils.metrics import calculate_metrics
from utils.feature_extraction import extract_clip_features_with_blocks

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP Feature Analysis')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/officehome',
                      help='Path to dataset directory')
    parser.add_argument('--features_file', type=str, default='clip_block_features_20250515_092209.pt',
                      help='Path to features file')
    parser.add_argument('--domains', nargs='+', default=None,
                      help='List of domains to analyze. If not provided, all domains in data_dir will be used.')
    
    # Feature extraction parameters
    parser.add_argument('--extract_features', action='store_true',
                      help='Extract CLIP features before analysis')
    parser.add_argument('--output_dir', type=str, default='data/features',
                      help='Directory to save extracted features')
    parser.add_argument('--model_name', type=str, default='ViT-B/16',
                      help='CLIP model name for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for computation')
    
    # Analysis parameters
    parser.add_argument('--analysis_type', type=str, choices=['single_class', 'all_classes', 'both'],
                      default='both', help='Type of analysis to perform')
    parser.add_argument('--class_idx', type=int, default=0,
                      help='Class index to analyze (for single_class analysis)')
    parser.add_argument('--max_samples', type=int, default=1000,
                      help='Maximum number of samples to use per class')
    parser.add_argument('--blocks', nargs='+', type=int, default=list(range(12)),
                      help='List of blocks to analyze')
    
    # Output parameters
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--timestamp', action='store_true',
                      help='Add timestamp to output filenames')
    
    return parser.parse_args()

def setup_directories(vis_dir):
    """Create necessary directories"""
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get image features
            image_features = model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features for all classes with improved prompts
            class_names = dataloader.dataset.classes
            # Replace underscores with spaces and add template
            prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
            text_tokens = clip.tokenize(prompts).to(device)
            text_features = model.get_text_features(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T)
            preds = similarity.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return calculate_metrics(all_preds, all_labels)

def get_original_model_results(args, datasets):
    """Get or load original model results."""
    original_results_file = os.path.join(args.output_dir, 'original_model_results.json')
    
    # Check if original results exist
    if os.path.exists(original_results_file):
        print("Loading original model results from file...")
        with open(original_results_file, 'r') as f:
            return json.load(f)
    
    print("Evaluating original model...")
    original_model = clip.load(args.model_name, device=args.device)[0]
    
    # Get model info
    model_info = {
        'model_name': args.model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Evaluate on each domain
    results = {}
    if isinstance(datasets, dict):
        for domain, dataset in datasets.items():
            print(f"\nEvaluating original model on {domain} domain...")
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            metrics = evaluate_model(original_model, dataloader, args.device)
            results[domain] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
    else:
        print("\nEvaluating original model on single domain...")
        dataloader = DataLoader(
            datasets,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        metrics = evaluate_model(original_model, dataloader, args.device)
        results['metrics'] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save original results
    original_results = {
        'results': results,
        'model_info': model_info,
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(original_results_file, 'w') as f:
        json.dump(original_results, f, indent=2)
    
    print(f"\nOriginal model results saved to {original_results_file}")
    return original_results

def get_dataset_name(data_dir):
    """Extract dataset name from data_dir path."""
    return os.path.basename(os.path.normpath(data_dir))

def get_domains(data_dir, domains_arg):
    """Get list of domains to process.
    
    Args:
        data_dir (str): Path to data directory
        domains_arg (list or None): List of domains provided by user, or None
    
    Returns:
        list: List of domain names to process
    """
    if domains_arg is not None and len(domains_arg) > 0:
        return domains_arg
    
    # If no domains specified, detect all subfolders as domains
    domains = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Auto-detected domains: {domains}")
    return domains

def main():
    args = parse_args()
    
    # Determine dataset name for subfolder organization
    dataset_name = get_dataset_name(args.data_dir)
    vis_dir = os.path.join(args.vis_dir, dataset_name)
    setup_directories(vis_dir)
    timestamp = get_timestamp() if args.timestamp else ""
    
    # Detect domains
    domains = get_domains(args.data_dir, args.domains)
    print(f"Domains: {domains}")
    
    # Load datasets
    datasets = load_datasets(args.data_dir, domains)
    
    # Extract features if requested
    if args.extract_features:
        # Use the new function from utils/feature_extraction
        _, features_file = extract_clip_features_with_blocks(
            data_dir=args.data_dir,
            domains=domains,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        # Determine features file path
        if os.path.isabs(args.features_file) or os.path.exists(args.features_file):
            features_file = args.features_file
        else:
            features_file = os.path.join('data', 'features', dataset_name, args.features_file)
        print(f"Loading features from {features_file}")
    
    features = torch.load(features_file)
    classnames = get_classnames_from_folder(args.data_dir, domains[0])
    
    print("\nAvailable classes:")
    for idx, name in enumerate(classnames):
        print(f"{idx}: {name}")
    
    # Perform requested analysis
    if args.analysis_type in ['single_class', 'both']:
        class_name = classnames[args.class_idx]
        print(f"\nAnalyzing single class: {class_name}")
        
        # Plot by domain
        save_path = os.path.join(vis_dir, f"tsne_multiblock_domain_{class_name}_{timestamp}.png")
        plot_tsne_multi_blocks(features, args.blocks, class_name, domains, save_path)
        print(f"Single class plot (by domain) saved to: {os.path.abspath(save_path)}")
    
    if args.analysis_type in ['all_classes', 'both']:
        print("\nAnalyzing all classes")
        
        # Plot by class
        save_path_class = os.path.join(vis_dir, f"tsne_multiblock_byclass_{timestamp}.png")
        plot_tsne_multi_blocks_by_class(features, args.blocks, domains, classnames,
                                       args.max_samples, save_path_class, color_by_domain=False)
        print(f"All classes plot (by class) saved to: {os.path.abspath(save_path_class)}")
        
        # Plot by domain
        save_path_domain = os.path.join(vis_dir, f"tsne_multiblock_bydomain_{timestamp}.png")
        plot_tsne_multi_blocks_by_class(features, args.blocks, domains, classnames,
                                       args.max_samples, save_path_domain, color_by_domain=True)
        print(f"All classes plot (by domain) saved to: {os.path.abspath(save_path_domain)}")

if __name__ == "__main__":
    main() 