import torch
import torch.nn as nn

def freeze_encoder(model: nn.Module, prefixes=["enc", "features", "backbone", "body"]):
    """
    Freeze all layers in the model except those matching the head prefixes.
    Actually, it's safer to freeze everything first, then unfreeze the head.
    
    Args:
        model: The PyTorch model
        prefixes: List of attribute names to strictly freeze (deprecated approach, 
                 we will freeze everything not in the head)
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    print("✓ Model encoder frozen.")

def unfreeze_head(model: nn.Module, head_prefixes=["head", "classifier", "fc", "fusion_layer"]):
    """
    Unfreeze layers that correspond to the classifier head.
    
    Args:
        model: The PyTorch model
        head_prefixes: List of potential names for the classifier head
    """
    # 2. Identify and unfreeze head
    unfrozen_count = 0
    all_param_names = [n for n,p in model.named_parameters()]
    
    for name, param in model.named_parameters():
        # Check if this parameter belongs to a head layer
        is_head = any(prefix in name for prefix in head_prefixes)
        
        if is_head:
            param.requires_grad = True
            unfrozen_count += 1
            
    if unfrozen_count == 0:
        print(f"⚠️ WARNING: No head layers found matching {head_prefixes}. Model is fully frozen!")
        print(f"Available layers: {all_param_names[:5]}...")
    else:
        print(f"✓ Unfrozen {unfrozen_count} head parameters for personalization.")
