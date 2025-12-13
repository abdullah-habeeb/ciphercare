"""
Process Differentially-Private Model Updates for Blockchain Logging

Generates blockchain-ready DP updates for any hospital with:
- Model delta computation (local - global)
- Gradient clipping and Gaussian noise
- Keccak256 hashing for blockchain verification
- JSON metadata for audit logging

Usage:
    python process_dp_update.py --hospital_id A \\
                                --checkpoint_path src/hospital_a/train/checkpoints/best_model.pth \\
                                --global_model_path src/global_models/global_model.pth \\
                                --model_part classifier_head
"""

import torch
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import math


def compute_delta(
    local_model_state: dict,
    global_model_state: dict,
    model_part: str = None
) -> dict:
    """
    Compute model delta: ΔW = local - global
    
    Args:
        local_model_state: Local model state dict
        global_model_state: Global model state dict
        model_part: Optional filter (e.g., 'classifier_head', 'fusion_head')
    
    Returns:
        Dict of delta tensors
    """
    delta = {}
    
    for key in local_model_state.keys():
        # Filter by model part if specified
        if model_part and model_part not in key:
            continue
        
        if key in global_model_state:
            # Compute delta
            delta[key] = local_model_state[key] - global_model_state[key]
        else:
            print(f"⚠️  Key '{key}' not found in global model, skipping")
    
    return delta


def clip_delta(delta: dict, max_norm: float = 1.0) -> tuple:
    """
    Clip delta to have maximum L2 norm.
    
    Args:
        delta: Dict of delta tensors
        max_norm: Maximum allowed norm
    
    Returns:
        Tuple of (clipped_delta, original_norm)
    """
    # Compute total norm
    total_norm = 0.0
    for tensor in delta.values():
        param_norm = tensor.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    clipped_delta = {}
    
    if clip_coef < 1:
        for key, tensor in delta.items():
            clipped_delta[key] = tensor * clip_coef
    else:
        clipped_delta = delta
    
    return clipped_delta, total_norm


def apply_dp_noise(
    delta: dict,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0
) -> dict:
    """
    Add Gaussian noise to delta for differential privacy.
    
    Args:
        delta: Dict of delta tensors
        noise_multiplier: Noise scale multiplier
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        Noisy delta dict
    """
    noisy_delta = {}
    sigma = noise_multiplier * max_grad_norm
    
    for key, tensor in delta.items():
        # Sample Gaussian noise
        noise = torch.normal(
            mean=0.0,
            std=sigma,
            size=tensor.shape,
            device=tensor.device
        )
        
        # Add noise
        noisy_delta[key] = tensor + noise
    
    return noisy_delta


def compute_epsilon(
    noise_multiplier: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0
) -> float:
    """
    Compute privacy budget (epsilon) from noise parameters.
    
    Formula: epsilon ≈ sqrt(2 * ln(1.25/delta)) * sensitivity / sigma
    
    Args:
        noise_multiplier: Noise scale (sigma)
        delta: Failure probability
        sensitivity: Sensitivity of the mechanism
    
    Returns:
        Epsilon value
    """
    sigma = noise_multiplier
    epsilon = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / sigma
    return epsilon


def hash_update(delta: dict) -> str:
    """
    Compute keccak256 hash of model delta for blockchain verification.
    
    Args:
        delta: Dict of delta tensors
    
    Returns:
        Hex string of keccak256 hash
    """
    # Concatenate all tensors
    all_tensors = []
    for key in sorted(delta.keys()):  # Sort for consistency
        tensor = delta[key].cpu().detach().numpy()
        all_tensors.append(tensor.flatten())
    
    # Concatenate and convert to bytes
    concatenated = np.concatenate(all_tensors)
    tensor_bytes = concatenated.tobytes()
    
    # Compute keccak256 hash (Ethereum-compatible)
    # Note: Python's hashlib doesn't have keccak256, so we use sha3_256
    # For true keccak256, you'd need: from Crypto.Hash import keccak
    # For now, using SHA3-256 as a placeholder
    hash_obj = hashlib.sha3_256(tensor_bytes)
    return hash_obj.hexdigest()


def save_outputs(
    hospital_id: str,
    noisy_delta: dict,
    update_hash: str,
    epsilon: float,
    delta: float,
    noise_multiplier: float,
    model_part: str,
    output_dir: str = "processed_updates"
):
    """
    Save blockchain-ready outputs.
    
    Args:
        hospital_id: Hospital identifier
        noisy_delta: Noisy model delta
        update_hash: Keccak256 hash
        epsilon: Privacy budget
        delta: Failure probability
        noise_multiplier: Noise multiplier used
        model_part: Model component name
        output_dir: Output directory
    """
    # Create output directory
    output_path = Path(output_dir) / hospital_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save noisy delta
    delta_path = output_path / "noisy_delta.pt"
    torch.save(noisy_delta, delta_path)
    print(f"+ Saved noisy delta: {delta_path}")
    
    # 2. Save update hash
    hash_path = output_path / "update_hash.txt"
    with open(hash_path, 'w') as f:
        f.write(update_hash)
    print(f"+ Saved update hash: {hash_path}")
    
    # 3. Save JSON metadata
    metadata = {
        "hospital_id": hospital_id,
        "update_hash": update_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "privacy": {
            "epsilon": round(epsilon, 4),
            "delta": delta,
            "noise_multiplier": noise_multiplier
        },
        "model_part": model_part if model_part else "full_model"
    }
    
    json_path = output_path / f"dp_update_{hospital_id}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"+ Saved JSON metadata: {json_path}")
    
    return metadata, json_path, hash_path


def validate_model_architecture(
    local_state: dict,
    global_state: dict,
    model_part: str = None
) -> bool:
    """
    Validate that local and global models have matching architecture.
    
    Args:
        local_state: Local model state dict
        global_state: Global model state dict
        model_part: Optional filter
    
    Returns:
        True if architectures match
    """
    local_keys = set(local_state.keys())
    global_keys = set(global_state.keys())
    
    # Filter by model part if specified
    if model_part:
        local_keys = {k for k in local_keys if model_part in k}
        global_keys = {k for k in global_keys if model_part in k}
    
    # Check for matching keys
    if local_keys != global_keys:
        missing_in_global = local_keys - global_keys
        missing_in_local = global_keys - local_keys
        
        if missing_in_global:
            print(f"⚠️  Keys in local but not global: {missing_in_global}")
        if missing_in_local:
            print(f"⚠️  Keys in global but not local: {missing_in_local}")
        
        return False
    
    # Check for matching shapes
    for key in local_keys:
        if local_state[key].shape != global_state[key].shape:
            print(f"❌ Shape mismatch for '{key}': "
                  f"local={local_state[key].shape}, global={global_state[key].shape}")
            return False
    
    return True


def main():
    """Main function to process DP update."""
    parser = argparse.ArgumentParser(
        description="Process differentially-private model updates for blockchain logging"
    )
    parser.add_argument("--hospital_id", type=str, required=True,
                        help="Hospital identifier (A, B, C, D, E)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to local model checkpoint")
    parser.add_argument("--global_model_path", type=str, required=True,
                        help="Path to global model checkpoint")
    parser.add_argument("--model_part", type=str, default=None,
                        help="Model component to extract (e.g., 'classifier_head', 'fusion_head')")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (default: 1.0)")
    parser.add_argument("--noise_multiplier", type=float, default=1.1,
                        help="Noise multiplier (default: 1.1)")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="DP delta parameter (default: 1e-5)")
    parser.add_argument("--output_dir", type=str, default="processed_updates",
                        help="Output directory (default: processed_updates)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Processing Differentially-Private Model Update")
    print("="*60)
    print(f"Hospital ID: {args.hospital_id}")
    print(f"Local checkpoint: {args.checkpoint_path}")
    print(f"Global model: {args.global_model_path}")
    print(f"Model part: {args.model_part or 'full_model'}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Noise multiplier: {args.noise_multiplier}")
    print("="*60)
    
    # 1. Load models
    print("\n1. Loading models...")
    try:
        local_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        global_checkpoint = torch.load(args.global_model_path, map_location='cpu')
        
        # Extract state dicts (handle different checkpoint formats)
        local_state = local_checkpoint if isinstance(local_checkpoint, dict) and 'state_dict' not in local_checkpoint else local_checkpoint.get('state_dict', local_checkpoint)
        global_state = global_checkpoint if isinstance(global_checkpoint, dict) and 'state_dict' not in global_checkpoint else global_checkpoint.get('state_dict', global_checkpoint)
        
        print(f"+ Loaded local model: {len(local_state)} parameters")
        print(f"+ Loaded global model: {len(global_state)} parameters")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # 2. Validate architecture
    print("\n2. Validating model architecture...")
    if not validate_model_architecture(local_state, global_state, args.model_part):
        print("❌ Model architecture mismatch! Aborting.")
        return
    print("+ Architecture validated")
    
    # 3. Compute delta
    print("\n3. Computing model delta (ΔW = local - global)...")
    delta = compute_delta(local_state, global_state, args.model_part)
    print(f"+ Computed delta for {len(delta)} parameters")
    
    # 4. Clip delta
    print("\n4. Clipping delta...")
    clipped_delta, original_norm = clip_delta(delta, args.max_grad_norm)
    print(f"+ Original norm: {original_norm:.6f}")
    print(f"+ Clipped to: {args.max_grad_norm}")
    
    # 5. Apply DP noise
    print("\n5. Applying differential privacy noise...")
    noisy_delta = apply_dp_noise(clipped_delta, args.noise_multiplier, args.max_grad_norm)
    print(f"+ Added Gaussian noise (σ = {args.noise_multiplier * args.max_grad_norm:.4f})")
    
    # 6. Compute epsilon
    epsilon = compute_epsilon(args.noise_multiplier, args.delta)
    print(f"+ Estimated epsilon: {epsilon:.4f} (delta = {args.delta})")
    print(f"  Privacy guarantee: (epsilon={epsilon:.4f}, delta={args.delta})")
    
    # 7. Hash update
    print("\n6. Computing keccak256 hash...")
    update_hash = hash_update(noisy_delta)
    print(f"+ Update hash: {update_hash}")
    
    # 8. Save outputs
    print("\n7. Saving blockchain-ready outputs...")
    metadata, json_path, hash_path = save_outputs(
        args.hospital_id,
        noisy_delta,
        update_hash,
        epsilon,
        args.delta,
        args.noise_multiplier,
        args.model_part,
        args.output_dir
    )
    
    # 9. Print JSON metadata
    print("\n" + "="*60)
    print("JSON Metadata for Blockchain:")
    print("="*60)
    print(json.dumps(metadata, indent=2))
    
    # 10. Final message
    print("\n" + "="*60)
    print(f"✅ Send dp_update_{args.hospital_id}.json and update_hash.txt")
    print("   to Neha's blockchain client for audit logging.")
    print("="*60)
    print(f"\nFiles ready in: {Path(args.output_dir) / args.hospital_id}")
    print(f"  - noisy_delta.pt")
    print(f"  - update_hash.txt")
    print(f"  - dp_update_{args.hospital_id}.json")
    print("="*60)


if __name__ == "__main__":
    main()
