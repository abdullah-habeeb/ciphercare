import torch
import os

ckpt_path = "src/hospital_a/train/checkpoints/best_model.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Checkpoint Keys:", len(ckpt.keys()))
    
    # Check specific layer to guess channels
    # 'residual_layer.residual_blocks.0.layer.linear.weight' or similar S4/Conv params
    
    # Check init_conv
    if 'init_conv.0.conv.weight' in ckpt:
        w = ckpt['init_conv.0.conv.weight']
        print(f"init_conv.0.conv.weight: {w.shape}") 
        # [res_channels, in_channels, 1] -> [C, 8, 1]
        
    # Check S4 specific params
    keys_to_check = [
        'residual_layer.residual_blocks.0.S41.s4_layer.kernel.kernel.P',
        'residual_layer.residual_blocks.0.S41.s4_layer.kernel.kernel.B',
        'residual_layer.residual_blocks.0.S41.s4_layer.kernel.kernel.C',
        'residual_layer.residual_blocks.0.S41.s4_layer.kernel.kernel.log_dt',
        'residual_layer.residual_blocks.0.S41.s4_layer.output_linear.weight'
    ]
    
    for k in keys_to_check:
        if k in ckpt:
            print(f"{k}: {ckpt[k].shape}")
        else:
            print(f"{k}: Not found")
            # Try to find partial match
            for ck in ckpt.keys():
                if 'kernel.P' in ck:
                    print(f"Found related: {ck}: {ckpt[ck].shape}")
                    break
else:
    print("Checkpoint not found")
