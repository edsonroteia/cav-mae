from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

# Importing the provided classes and functions from modeling_finetune.py
from models.modeling_pretrain import pretrain_videomae_small_patch16_224

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

# Define the dummy input creation and forward pass
def main():
    # Instantiate the model using the provided function
    model = pretrain_videomae_small_patch16_224(pretrained=False)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 8, 224, 224)  # Batch size 1, 3 color channels, 224x224 image size
    mask = torch.zeros((1, 196 * 4), dtype=torch.bool)  # Dummy mask (all patches are visible)
    
    # Perform a forward pass
    output = model(dummy_input, mask)
    
    # Print the output shape
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()