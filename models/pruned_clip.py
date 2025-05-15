import torch
import torch.nn as nn
import clip
from typing import List, Optional, Union, Dict

class PrunedCLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B/16",
        prune_vision_layers: Optional[List[int]] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True
    ):
        """
        Initialize a pruned CLIP model with vision encoder pruning.
        
        Args:
            model_name: Name of the CLIP model to use (e.g., "ViT-B/16", "ViT-B/32")
            prune_vision_layers: List of layer indices to prune from vision encoder (0-based).
                               If None, no layers are pruned.
            freeze_vision: Whether to freeze the vision encoder
            freeze_text: Whether to freeze the text encoder
        """
        super().__init__()
        
        # Load base CLIP model
        self.clip_model, self.preprocess = clip.load(model_name, device='cpu')
        self.model_name = model_name
        
        # Store original number of layers
        self.num_vision_layers = len(self.clip_model.visual.transformer.resblocks)
        self.num_text_layers = len(self.clip_model.transformer.resblocks)
        
        # Store original layer dimensions
        self.hidden_size = self.clip_model.visual.transformer.width
        
        # Apply vision encoder pruning if specified
        if prune_vision_layers is not None:
            self._prune_vision_layers(prune_vision_layers)
        
        # Freeze encoders if specified
        if freeze_vision:
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
                
        if freeze_text:
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False
    
    @classmethod
    def get_original_clip(cls, model_name: str = "ViT-B/16") -> 'PrunedCLIP':
        """
        Get an instance of the original CLIP model without any pruning.
        
        Args:
            model_name: Name of the CLIP model to use
            
        Returns:
            PrunedCLIP instance with no pruning
        """
        return cls(model_name=model_name, prune_vision_layers=None)
    
    def _get_model_config(self, model_name: str) -> Dict:
        """
        Get model configuration based on model name.
        
        Args:
            model_name: Name of the CLIP model (e.g., "ViT-B/16", "ViT-B/32")
            
        Returns:
            Dictionary containing model configuration
        """
        # Default configuration
        config = {
            'input_resolution': 224,  # CLIP uses 224x224 input resolution
            'patch_size': 16,  # Default to 16
            'width': 768,  # Default width for ViT-B
            'layers': 12,  # Default number of layers for ViT-B
            'heads': 12,  # Default number of heads for ViT-B
        }
        
        # Update patch size based on model name
        if model_name.endswith('/32'):
            config['patch_size'] = 32
        
        return config
    
    def _prune_vision_layers(self, prune_layers: List[int]):
        """
        Prune specified layers from the vision encoder by creating a new transformer
        with only the non-pruned layers.
        
        Args:
            prune_layers: List of layer indices to prune (0-based)
        """
        # Sort and validate prune layers
        prune_layers = sorted(prune_layers)
        if max(prune_layers) >= self.num_vision_layers:
            raise ValueError(f"Prune layer indices must be less than {self.num_vision_layers}")
        
        # Get the original transformer configuration
        original_transformer = self.clip_model.visual.transformer
        original_visual = self.clip_model.visual
        
        # Get model configuration
        config = self._get_model_config(self.model_name)
        
        # Get output dimension from the visual model
        # In CLIP, the output dimension is the same as the transformer's width
        output_dim = original_transformer.width
        
        # Create new transformer with fewer layers
        new_transformer = clip.model.VisionTransformer(
            input_resolution=config['input_resolution'],
            patch_size=config['patch_size'],
            width=original_transformer.width,
            layers=original_transformer.layers - len(prune_layers),
            heads=original_transformer.resblocks[0].attn.num_heads,
            output_dim=output_dim
        )
        
        # Copy weights from non-pruned layers
        new_layer_idx = 0
        for old_layer_idx in range(self.num_vision_layers):
            if old_layer_idx not in prune_layers:
                # Get the original and new transformer blocks
                old_block = original_transformer.resblocks[old_layer_idx]
                new_block = new_transformer.transformer.resblocks[new_layer_idx]
                
                # Copy the weights
                new_block.load_state_dict(old_block.state_dict())
                new_layer_idx += 1
        
        # Copy the patch embedding and positional encoding
        new_transformer.conv1.load_state_dict(original_visual.conv1.state_dict())
        with torch.no_grad():
            new_transformer.class_embedding.data = original_visual.class_embedding.data.clone()
            new_transformer.positional_embedding.data = original_visual.positional_embedding.data.clone()
            new_transformer.proj.data = original_visual.proj.data.clone()
        new_transformer.ln_pre.load_state_dict(original_visual.ln_pre.state_dict())
        new_transformer.ln_post.load_state_dict(original_visual.ln_post.state_dict())
        
        # Replace the original transformer with the pruned one
        self.clip_model.visual = new_transformer
        
        # Update the number of layers
        self.num_vision_layers = len(new_transformer.transformer.resblocks)
        
        # Store pruning information
        self.kept_layers = [i for i in range(self.num_vision_layers) if i not in prune_layers]
        self.pruned_layers = prune_layers
    
    def get_vision_encoder_info(self) -> Dict:
        """
        Get information about the vision encoder structure.
        
        Returns:
            Dictionary containing information about the vision encoder
        """
        return {
            "model_name": self.model_name,
            "total_layers": self.num_vision_layers,
            "hidden_size": self.hidden_size,
            "kept_layers": self.kept_layers if hasattr(self, 'kept_layers') else list(range(self.num_vision_layers)),
            "pruned_layers": self.pruned_layers if hasattr(self, 'pruned_layers') else [],
            "layer_structure": [
                {
                    "layer_idx": i,
                    "layer_type": type(layer).__name__,
                    "is_pruned": i in getattr(self, 'pruned_layers', []),
                    "hidden_size": self.hidden_size
                }
                for i, layer in enumerate(self.clip_model.visual.transformer.resblocks)
            ]
        }
    
    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Forward pass through the pruned CLIP model.
        """
        return self.clip_model(image, text)
    
    def get_image_features(self, pixel_values: torch.FloatTensor, return_all_blocks: bool = False) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Get image features from the vision encoder.
        
        Args:
            pixel_values: Input images tensor
            return_all_blocks: If True, returns features from all transformer blocks
            
        Returns:
            If return_all_blocks is False, returns the final image features.
            If return_all_blocks is True, returns a list of features from all blocks.
        """
        if not return_all_blocks:
            return self.clip_model.encode_image(pixel_values)
        
        # Get features from all blocks
        x = self.clip_model.visual.conv1(pixel_values)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Store features from each block
        block_features = []
        for block in self.clip_model.visual.transformer.resblocks:
            x = block(x)
            # Get the CLS token features (first token)
            block_features.append(x[0].permute(1, 0))  # shape = [batch_size, width]
        
        # Add the final layer norm
        x = self.clip_model.visual.ln_post(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        block_features.append(x[0])  # Add final features
        
        return block_features
    
    def get_text_features(
        self,
        text: torch.LongTensor
    ) -> torch.FloatTensor:
        """Get text features from the text encoder."""
        return self.clip_model.encode_text(text)
    
    def zero_shot_classify(
        self,
        image: torch.FloatTensor,
        class_names: List[str],
        temperature: float = 100.0
    ) -> Dict[str, float]:
        """
        Perform zero-shot classification using the pruned CLIP model.
        
        Args:
            image: Input image tensor
            class_names: List of class names to classify against
            temperature: Temperature for softmax scaling
            
        Returns:
            Dictionary mapping class names to their probabilities
        """
        # Get image features
        image_features = self.get_image_features(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get text features for each class
        text_tokens = clip.tokenize(class_names).to(image.device)
        text_features = self.get_text_features(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Convert to dictionary
        return {name: prob.item() for name, prob in zip(class_names, similarity[0])} 