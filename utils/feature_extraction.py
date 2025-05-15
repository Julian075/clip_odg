import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from utils.dataset import get_classnames_from_folder
import clip

class BlockFeatureCollector:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        for i, block in enumerate(self.model.visual.transformer.resblocks):
            self.hooks.append(block.register_forward_hook(
                lambda m, inp, out, idx=i: self._hook_fn(m, inp, out, idx)
            ))
        self._current_labels = None
        self._current_domain = None
        self._current_classnames = None

    def set_batch_info(self, labels, domain, classnames):
        self._current_labels = labels
        self._current_domain = domain
        self._current_classnames = classnames

    def _hook_fn(self, module, input, output, block_idx):
        cls_token = output[0, :, :].detach().cpu()
        batch_size = len(self._current_labels)
        
        if block_idx not in self.features:
            self.features[block_idx] = {}
        domain = self._current_domain
        if domain not in self.features[block_idx]:
            self.features[block_idx][domain] = {}
        labels = self._current_labels
        classnames = self._current_classnames
        
        for i in range(batch_size):
            label_idx = int(labels[i].item())
            class_name = classnames[label_idx]
            if class_name not in self.features[block_idx][domain]:
                self.features[block_idx][domain][class_name] = []
            self.features[block_idx][domain][class_name].append(cls_token[i])

    def clear(self):
        self.features = {}

    def get_features(self):
        return self.features

def extract_clip_features_with_blocks(data_dir, domains, model_name, batch_size, device, output_dir=None):
    """
    Extract CLIP features from all transformer blocks for all images in the dataset.
    Args:
        data_dir (str): Path to dataset directory
        domains (list): List of domains to process
        model_name (str): CLIP model name
        batch_size (int): Batch size
        device (str): Device to use
        output_dir (str or None): Directory to save features (if None, does not save)
    Returns:
        all_features (dict): Nested dict of features by block, domain, and class
        save_path (str or None): Path to saved features file (if output_dir is not None)
    """
    from PIL import Image
    domain_for_classnames = domains[0]
    classnames = get_classnames_from_folder(data_dir, domain_for_classnames)
    print("Class index to name mapping:")
    for idx, name in enumerate(classnames):
        print(f"{idx}: {name}")
    
    # Load CLIP model
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    collector = BlockFeatureCollector(model)
    
    # Prepare text features
    prompts = [f"A photo of a {name}." for name in classnames]
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Helper to get dataset
    def get_dataset(data_dir, domain, classnames):
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classnames)}
        samples = []
        for cls in classnames:
            class_dir = os.path.join(data_dir, domain, cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(class_dir, img_name), class_to_idx[cls]))
        return samples
    
    # Extract features and compute accuracy for each domain
    all_features = {}
    for domain in domains:
        print(f"\nProcessing domain: {domain}")
        samples = get_dataset(data_dir, domain, classnames)
        all_preds = []
        all_labels = []
        collector.clear()
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            images = []
            labels = []
            for img_path, label in batch:
                img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
                images.append(img)
                labels.append(label)
            images = torch.cat(images, dim=0).to(device)
            labels = torch.tensor(labels)
            
            with torch.no_grad():
                collector.set_batch_info(labels, domain, classnames)
                image_features = model.encode_image(images).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Save features for this domain
        features = collector.get_features()
        for block, dom_dict in features.items():
            if block not in all_features:
                all_features[block] = {}
            for dom, class_dict in dom_dict.items():
                if dom not in all_features[block]:
                    all_features[block][dom] = {}
                for cls, embs in class_dict.items():
                    if cls not in all_features[block][dom]:
                        all_features[block][dom][cls] = []
                    all_features[block][dom][cls].extend(embs)
        
        # Compute accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        print(f"Domain {domain} accuracy: {acc*100:.2f}% ({len(all_labels)} samples)")
    
    # Print summary
    print("\nSummary of collected features:")
    for block, dom_dict in all_features.items():
        for dom, class_dict in dom_dict.items():
            for cls, embs in class_dict.items():
                print(f"Block {block} | Domain {dom} | Class {cls}: {len(embs)} samples, emb shape: {embs[0].shape if embs else 'N/A'}")
    
    # Save features
    save_path = None
    if output_dir is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(output_dir, f"clip_block_features_{timestamp}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(all_features, save_path)
        print(f"\nSaved features to {save_path}")
    return all_features, save_path 