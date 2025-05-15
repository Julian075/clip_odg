import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class DomainDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            domain (str): Name of the domain (e.g., 'Art', 'Clipart', 'Product', 'Real_World')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, domain)
        self.domain = domain  # Store the domain name
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and their labels
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataset(data_dir, domain=None, transform=None):
    """
    Get dataset from data directory.
    
    Args:
        data_dir (str): Path to data directory
        domain (str, optional): Specific domain to load. If None, loads all domains.
        transform (callable, optional): Transform to apply to images.
    
    Returns:
        Dataset or dict of datasets
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if domain is not None:
        return DomainDataset(data_dir, domain, transform)
    else:
        domains = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return {domain: DomainDataset(data_dir, domain, transform) for domain in domains}

def get_classnames_from_folder(data_dir, domain):
    """Get class names from the dataset folder structure"""
    domain_path = os.path.join(data_dir, domain)
    classnames = sorted([d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))])
    return classnames

def load_datasets(data_dir, domains):
    """
    Load datasets for specified domains.
    
    Args:
        data_dir (str): Path to data directory
        domains (list): List of domain names to load
    
    Returns:
        dict: Dictionary mapping domain names to datasets
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {domain: DomainDataset(data_dir, domain, transform) for domain in domains} 