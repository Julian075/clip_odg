import os
import torch
import numpy as np
import clip
from utils.dataset import get_classnames_from_folder

def load_features_pt(pt_path):
    return torch.load(pt_path)

def gather_features_and_labels(features, block_idx, domains):
    all_features = []
    all_labels = []
    domain_labels = []
    class_names = sorted(list(features[block_idx][domains[0]].keys()))  # Get class names from first domain
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for domain in domains:
        if domain in features[block_idx]:
            for class_name in class_names:
                if class_name in features[block_idx][domain]:
                    feats = features[block_idx][domain][class_name]
                    all_features.extend([f.numpy() for f in feats])
                    all_labels.extend([class_to_idx[class_name]] * len(feats))
                    domain_labels.extend([domain] * len(feats))
    return np.stack(all_features), np.array(all_labels), np.array(domain_labels), class_names

def main():
    features_file = "clip_block_features_20250515_092209.pt"  # Updated features file
    data_dir = "data/officehome"
    domain_for_classnames = "art"
    domains = ['art', 'clipart', 'product', 'real_world']
    block_idx = 11  # Last block
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features and get class names
    features = load_features_pt(features_file)
    _, _, _, classnames = gather_features_and_labels(features, block_idx, domains)
    
    print("Class index to name mapping:")
    for idx, name in enumerate(classnames):
        print(f"{idx}: {name}")

    # Load CLIP model and get projection components
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    
    # Get the exact components from CLIP's visual encoder
    ln_post = model.visual.ln_post
    proj = model.visual.proj

    # Prepare text features
    prompts = [f"A photo of a {name}." for name in classnames]
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def process_features(feats):
        # Process features exactly like CLIP does
        feats = torch.tensor(feats, device=device)  # Keep original dtype
        with torch.no_grad():
            # Apply LayerNorm and projection exactly as CLIP does
            feats = ln_post(feats)  # [batch_size, 768]
            # Project to the same dimension as text features
            feats = feats @ proj  # [batch_size, 512]
            # Normalize features
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    # Accuracy per domain
    for domain in domains:
        print(f"\nEvaluating domain: {domain}")
        feats = []
        labels = []
        if domain in features[block_idx]:
            for class_name in classnames:
                if class_name in features[block_idx][domain]:
                    feats.extend([f.numpy() for f in features[block_idx][domain][class_name]])
                    labels.extend([classnames.index(class_name)] * len(features[block_idx][domain][class_name]))
        if not feats:
            print(f"No features for domain {domain}")
            continue
            
        # Process features
        feats = np.stack(feats)
        feats = process_features(feats)
        
        # Compute logits
        with torch.no_grad():
            logits = feats @ text_features.T
            preds = logits.argmax(dim=1).cpu().numpy()
            
        labels = np.array(labels)
        acc = (preds == labels).mean()
        print(f"Domain {domain} accuracy: {acc*100:.2f}% ({len(labels)} samples)")

    # Overall accuracy
    print("\nEvaluating all domains together...")
    all_feats, all_labels, _, _ = gather_features_and_labels(features, block_idx, domains)
    feats = process_features(all_feats)
    
    # Compute logits
    with torch.no_grad():
        logits = feats @ text_features.T
        preds = logits.argmax(dim=1).cpu().numpy()
        
    acc = (preds == all_labels).mean()
    print(f"Overall accuracy (all domains): {acc*100:.2f}% ({len(all_labels)} samples)")

if __name__ == "__main__":
    main() 