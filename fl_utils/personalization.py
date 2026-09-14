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

    print("+ Model encoder frozen.")

def unfreeze_head(model: nn.Module, head_prefixes=["head", "classifier", "fc", "fusion_layer"]):
    """
    Unfreeze layers that correspond to the classifier head.

    Args:
        model: The PyTorch model
        head_prefixes: List of potential names for the classifier head
    """
    # 2. Identify and unfreeze head
    unfrozen_count = 0
    named_params = list(model.named_parameters())

    for name, param in named_params:
        # Check if this parameter belongs to a head layer
        is_head = any(prefix in name for prefix in head_prefixes)

        if is_head:
            param.requires_grad = True
            unfrozen_count += 1

    if unfrozen_count == 0:
        # None of head_prefixes matched this model's naming (e.g. UnifiedFLModel
        # names its layers net.0..net.8, matching none of the prefixes above,
        # which used to leave the whole model frozen with 0 trainable params).
        # Fall back to unfreezing the last parameterized module, which is the
        # output layer for any plain feedforward/sequential architecture.
        last_top_level_name = named_params[-1][0].rsplit(".", 1)[0]
        for name, param in named_params:
            if name.rsplit(".", 1)[0] == last_top_level_name:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"+ No head layers matched {head_prefixes}; falling back to last "
              f"module '{last_top_level_name}'. Unfroze {unfrozen_count} parameters.")
    else:
        print(f"+ Unfrozen {unfrozen_count} head parameters for personalization.")
