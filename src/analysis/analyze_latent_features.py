import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import clip
from utils.dataset import get_dataset, get_classnames_from_folder

# ---- CONFIG ----
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-B/16'
DATA_DIR = 'data/officehome'
DOMAINS = ['art', 'clipart', 'product', 'real_world']

def get_classnames_from_folder(data_dir, domain):
    domain_path = os.path.join(data_dir, domain)
    classnames = sorted([d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))])
    return classnames

# ---- HOOKED FEATURE EXTRACTOR ----
class BlockFeatureCollector:
    def __init__(self, model):
        self.model = model
        # {block_idx: {domain: {class_name: [emb, ...]}}}
        self.features = {}
        self.hooks = []
        for i, block in enumerate(self.model.visual.transformer.resblocks):
            self.hooks.append(block.register_forward_hook(
                lambda m, inp, out, idx=i: self._hook_fn(m, inp, out, idx)
            ))
        self._current_labels = None
        self._current_domain = None
        self._current_classnames = None
        self._printed_shapes = False

    def set_batch_info(self, labels, domain, classnames):
        self._current_labels = labels
        self._current_domain = domain
        self._current_classnames = classnames

    def _hook_fn(self, module, input, output, block_idx):
        # output shape: (batch_size, 197, 768)
        # - batch_size: number of images in the batch
        # - 197: number of tokens per image (196 patches + 1 [CLS])
        # - 768: embedding dimension
        
        # Extract only the [CLS] token for each image in the batch and move to CPU immediately
        cls_token = output[0, :, :].detach().cpu()  # shape: (batch_size, 768)
        batch_size = len(self._current_labels)
        
        # Debug prints for first batch
        if not self._printed_shapes:
            print(f"\nBlock {block_idx} shapes:")
            print(f"  - output shape: {output.shape}  # (batch_size, 197, 768)")
            print(f"  - cls_token shape: {cls_token.shape}  # (batch_size, 768)")
            print(f"  - labels shape: {self._current_labels.shape}  # (batch_size,)")
            print(f"  - batch_size: {batch_size}")
            print(f"  - cls_token dtype: {cls_token.dtype}")
        
        if block_idx not in self.features:
            self.features[block_idx] = {}
        domain = self._current_domain
        if domain not in self.features[block_idx]:
            self.features[block_idx][domain] = {}
        labels = self._current_labels
        classnames = self._current_classnames
        
        # Store cls_token for each image in the batch
        for i in range(batch_size):
            label_idx = int(labels[i].item())  # Get class index for this image
            class_name = classnames[label_idx]  # Get class name from index
            if class_name not in self.features[block_idx][domain]:
                self.features[block_idx][domain][class_name] = []
            # Store the [CLS] token embedding for this image
            self.features[block_idx][domain][class_name].append(cls_token[i])  # shape: (768,)
        
        # Print shape after first batch only
        if not self._printed_shapes:
            print(f"Block {block_idx}: class token shape = {cls_token.shape[1:]}")
            self._printed_shapes = True

    def clear(self):
        self.features = {}
        self._printed_shapes = False

    def get_features(self):
        return self.features

# ---- MAIN SCRIPT ----
def main():
    print(f"Loading CLIP model: {MODEL_NAME}")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    collector = BlockFeatureCollector(model)

    # Get class names from art domain (they should be the same across domains)
    classnames = get_classnames_from_folder(DATA_DIR, DOMAINS[0])
    print("\nClass names:")
    for idx, name in enumerate(classnames):
        print(f"{idx}: {name}")

    all_features = {}  # {block: {domain: {class_name: [emb, ...]}}}
    for domain in DOMAINS:
        print(f"\nProcessing domain: {domain}")
        dataset = get_dataset(DATA_DIR, domain, preprocess)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        collector.clear()
        first_batch = True
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(DEVICE)
                collector.set_batch_info(labels, domain, classnames)
                _ = model.encode_image(images)
                if first_batch:
                    first_batch = False
        # Save features for this domain
        features = collector.get_features()  # {block: {domain: {class_name: [emb, ...]}}}
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

    # Print summary
    print("\nSummary of collected features:")
    for block, dom_dict in all_features.items():
        for dom, class_dict in dom_dict.items():
            for cls, embs in class_dict.items():
                print(f"Block {block} | Domain {dom} | Class {cls}: {len(embs)} samples, emb shape: {embs[0].shape if embs else 'N/A'}, dtype: {embs[0].dtype if embs else 'N/A'}")

    # Save to disk
    save_path = f"clip_block_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(all_features, save_path)
    print(f"\nSaved features to {save_path}")

if __name__ == "__main__":
    main() 