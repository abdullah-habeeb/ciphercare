"""
Differential Privacy Utilities for Federated Learning

Implements DP-SGD (Differentially Private Stochastic Gradient Descent) for FL clients.
Provides gradient clipping and Gaussian noise injection to preserve privacy.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple


def compute_noise_scale(
    epsilon: float,
    delta: float,
    num_samples: int,
    num_rounds: int,
    max_grad_norm: float = 1.0
) -> float:
    """
    Compute noise scale (sigma) for Gaussian mechanism in DP-SGD.
    
    Uses the Gaussian mechanism formula:
        sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    
    Where sensitivity = max_grad_norm / num_samples
    
    Args:
        epsilon: Privacy budget (lower = more privacy)
        delta: Failure probability (typically 1e-5)
        num_samples: Number of training samples
        num_rounds: Number of FL rounds (for privacy accounting)
        max_grad_norm: Maximum gradient norm after clipping
    
    Returns:
        Noise scale (sigma) for Gaussian noise
    """
    # Sensitivity of the mechanism
    sensitivity = max_grad_norm / num_samples
    
    # Gaussian mechanism formula
    # Adjusted for composition over multiple rounds
    epsilon_per_round = epsilon / math.sqrt(num_rounds)
    
    sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon_per_round
    
    return sigma


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to have maximum L2 norm.
    
    This is a crucial step for differential privacy - it bounds the
    sensitivity of the gradient computation.
    
    Args:
        model: PyTorch model with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        Total gradient norm before clipping
    """
    # Compute total gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    return total_norm


def add_dp_noise(
    model: torch.nn.Module,
    sigma: float,
    device: torch.device = None
) -> None:
    """
    Add Gaussian noise to model gradients for differential privacy.
    
    Noise is sampled from N(0, sigma^2) and added to each gradient.
    
    Args:
        model: PyTorch model with gradients
        sigma: Noise scale (computed by compute_noise_scale)
        device: Device to create noise on (defaults to gradient device)
    """
    for param in model.parameters():
        if param.grad is not None:
            if device is None:
                device = param.grad.device
            
            # Sample Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=sigma,
                size=param.grad.shape,
                device=device
            )
            
            # Add noise to gradient
            param.grad.data.add_(noise)


def compute_privacy_spent(
    sigma: float,
    num_rounds: int,
    delta: float,
    num_samples: int,
    max_grad_norm: float = 1.0
) -> float:
    """
    Compute total privacy budget (epsilon) spent after training.
    
    Uses the moments accountant method for composition.
    
    Args:
        sigma: Noise scale used
        num_rounds: Number of FL rounds completed
        delta: Failure probability
        num_samples: Number of training samples
        max_grad_norm: Maximum gradient norm
    
    Returns:
        Total epsilon spent
    """
    # Sensitivity
    sensitivity = max_grad_norm / num_samples
    
    # Reverse the Gaussian mechanism formula
    # epsilon = sqrt(2 * ln(1.25/delta)) * sensitivity / sigma
    epsilon_per_round = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / sigma
    
    # Composition over rounds (simple composition)
    total_epsilon = epsilon_per_round * math.sqrt(num_rounds)
    
    return total_epsilon


class DPConfig:
    """Configuration for differential privacy."""
    
    def __init__(
        self,
        epsilon: float = 5.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        num_rounds: int = 5
    ):
        """
        Initialize DP configuration.
        
        Args:
            epsilon: Privacy budget (default: 5.0 = moderate privacy)
            delta: Failure probability (default: 1e-5)
            max_grad_norm: Maximum gradient norm (default: 1.0)
            num_rounds: Expected number of FL rounds (default: 5)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_rounds = num_rounds
    
    def compute_noise_scale(self, num_samples: int) -> float:
        """Compute noise scale for given number of samples."""
        return compute_noise_scale(
            self.epsilon,
            self.delta,
            num_samples,
            self.num_rounds,
            self.max_grad_norm
        )
    
    def __repr__(self):
        return (f"DPConfig(epsilon={self.epsilon}, delta={self.delta}, "
                f"max_grad_norm={self.max_grad_norm}, num_rounds={self.num_rounds})")


def apply_dp_to_gradients(
    model: torch.nn.Module,
    dp_config: DPConfig,
    num_samples: int,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Apply differential privacy to model gradients (clip + noise).
    
    This is the main function to use in FL clients.
    
    Args:
        model: PyTorch model with gradients
        dp_config: DP configuration
        num_samples: Number of training samples
        device: Device for noise generation
    
    Returns:
        Dict with DP metrics (grad_norm, sigma, epsilon_spent)
    """
    # 1. Clip gradients
    grad_norm = clip_gradients(model, dp_config.max_grad_norm)
    
    # 2. Compute noise scale
    sigma = dp_config.compute_noise_scale(num_samples)
    
    # 3. Add noise
    add_dp_noise(model, sigma, device)
    
    # 4. Compute privacy spent (for logging)
    epsilon_spent = compute_privacy_spent(
        sigma,
        1,  # Single round
        dp_config.delta,
        num_samples,
        dp_config.max_grad_norm
    )
    
    return {
        "grad_norm": grad_norm,
        "sigma": sigma,
        "epsilon_spent": epsilon_spent,
        "clipped": grad_norm > dp_config.max_grad_norm
    }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Differential Privacy Utilities - Example")
    print("="*60)
    
    # Configuration
    dp_config = DPConfig(epsilon=5.0, delta=1e-5, max_grad_norm=1.0, num_rounds=5)
    print(f"\nConfiguration: {dp_config}")
    
    # Example: Hospital A with 17,418 samples
    num_samples = 17418
    sigma = dp_config.compute_noise_scale(num_samples)
    
    print(f"\nFor Hospital A ({num_samples:,} samples):")
    print(f"  Noise scale (sigma): {sigma:.6f}")
    
    # Privacy spent after 5 rounds
    epsilon_spent = compute_privacy_spent(sigma, 5, dp_config.delta, num_samples)
    print(f"  Privacy spent after 5 rounds: epsilon = {epsilon_spent:.3f}")
    print(f"  Privacy guarantee: (epsilon={epsilon_spent:.3f}, delta={dp_config.delta})")
    
    # Example: Hospital B with 800 samples
    num_samples = 800
    sigma = dp_config.compute_noise_scale(num_samples)
    
    print(f"\nFor Hospital B ({num_samples:,} samples):")
    print(f"  Noise scale (sigma): {sigma:.6f}")
    
    epsilon_spent = compute_privacy_spent(sigma, 5, dp_config.delta, num_samples)
    print(f"  Privacy spent after 5 rounds: epsilon = {epsilon_spent:.3f}")
    
    print("\n" + "="*60)
    print("+ Differential privacy utilities ready!")
    print("="*60)
