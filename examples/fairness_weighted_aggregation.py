"""
Example: Using Domain Relevance Scoring in FL Aggregation

This demonstrates how to integrate the automatic domain relevance scoring
into the federated learning server's fairness-weighted aggregation.
"""

import json
import sys
from typing import Dict, Tuple, List

sys.path.append('.')
from fl_utils.domain_relevance import load_hospital_profiles, compute_domain_relevance_matrix



def compute_fairness_weights(
    hospital_id: str,
    auroc: float,
    num_samples: int,
    total_samples: int,
    domain_relevance_matrix: Dict[Tuple[str, str], float],
    target_hospital: str = None,
    auroc_weight: float = 0.6,
    sample_weight: float = 0.3,
    relevance_weight: float = 0.1
) -> float:
    """
    Compute fairness weight for a hospital in FL aggregation.
    
    Formula:
        weight = 0.6 * AUROC² + 0.3 * (samples / total_samples) + 0.1 * domain_relevance
    
    Args:
        hospital_id: ID of the hospital
        auroc: Hospital's AUROC score
        num_samples: Number of samples at this hospital
        total_samples: Total samples across all hospitals
        domain_relevance_matrix: Pre-computed domain relevance scores
        target_hospital: Target hospital for relevance (if None, use average relevance)
        auroc_weight: Weight for AUROC component (default: 0.6)
        sample_weight: Weight for sample count component (default: 0.3)
        relevance_weight: Weight for domain relevance component (default: 0.1)
    
    Returns:
        Fairness weight for this hospital
    """
    # Component 1: AUROC squared (rewards high performance)
    auroc_component = auroc_weight * (auroc ** 2)
    
    # Component 2: Normalized sample count (rewards larger datasets)
    sample_component = sample_weight * (num_samples / total_samples)
    
    # Component 3: Domain relevance (rewards relevant hospitals)
    if target_hospital:
        relevance_score = domain_relevance_matrix.get((hospital_id, target_hospital), 0.3)
    else:
        # Use average relevance to all other hospitals
        relevances = [
            score for (h1, h2), score in domain_relevance_matrix.items()
            if h1 == hospital_id and h2 != hospital_id
        ]
        relevance_score = sum(relevances) / len(relevances) if relevances else 0.5
    
    relevance_component = relevance_weight * relevance_score
    
    # Final weight
    weight = auroc_component + sample_component + relevance_component
    
    return weight


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0.
    
    Args:
        weights: Dict mapping hospital_id to weight
    
    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        # Equal weights if all zero
        return {h: 1.0 / len(weights) for h in weights}
    
    return {h: w / total for h, w in weights.items()}


def main():
    """
    Example: Compute fairness weights for FL Round 1
    """
    print("="*60)
    print("Example: Fairness-Weighted FL Aggregation")
    print("="*60)
    
    # 1. Load domain relevance matrix
    print("\n1. Loading domain relevance matrix...")
    hospitals, _ = load_hospital_profiles("fl_config/hospital_profiles.json")
    domain_relevance_matrix = compute_domain_relevance_matrix(hospitals)
    print(f"   ✓ Loaded relevance scores for {len(hospitals)} hospitals")
    
    # 2. Simulate FL Round 1 results
    print("\n2. Simulating FL Round 1 results...")
    fl_round_results = {
        "A": {"auroc": 0.72, "samples": 17418},
        "B": {"auroc": 0.96, "samples": 800},
        "C": {"auroc": 0.65, "samples": 160},
        "D": {"auroc": 0.68, "samples": 2400},
        "E": {"auroc": 0.75, "samples": 2400}
    }
    
    total_samples = sum(r["samples"] for r in fl_round_results.values())
    
    # 3. Compute fairness weights
    print("\n3. Computing fairness weights...")
    print(f"   Formula: 0.6*AUROC² + 0.3*samples + 0.1*domain_relevance")
    print()
    
    weights = {}
    for hospital_id, results in fl_round_results.items():
        weight = compute_fairness_weights(
            hospital_id=hospital_id,
            auroc=results["auroc"],
            num_samples=results["samples"],
            total_samples=total_samples,
            domain_relevance_matrix=domain_relevance_matrix,
            target_hospital=None  # Use average relevance
        )
        weights[hospital_id] = weight
    
    # 4. Normalize weights
    normalized_weights = normalize_weights(weights)
    
    # 5. Display results
    print("="*60)
    print("Fairness Weights for FL Round 1")
    print("="*60)
    print(f"{'Hospital':<12} {'AUROC':<8} {'Samples':<10} {'Weight':<10} {'Normalized':<12}")
    print("-"*60)
    
    for hospital_id in sorted(fl_round_results.keys()):
        auroc = fl_round_results[hospital_id]["auroc"]
        samples = fl_round_results[hospital_id]["samples"]
        weight = weights[hospital_id]
        norm_weight = normalized_weights[hospital_id]
        
        print(f"{hospital_id:<12} {auroc:<8.3f} {samples:<10,} {weight:<10.4f} {norm_weight:<12.4f}")
    
    print("="*60)
    
    # 6. Show impact of domain relevance
    print("\n4. Domain Relevance Impact:")
    print("   (How much each hospital contributes to others)")
    print()
    
    for target_id in ["A", "D", "E"]:
        print(f"   Target: Hospital {target_id}")
        for source_id in fl_round_results.keys():
            if source_id != target_id:
                relevance = domain_relevance_matrix.get((source_id, target_id), 0.0)
                print(f"     - Hospital {source_id} → {target_id}: {relevance:.3f}")
        print()
    
    # 7. Save example output
    output = {
        "round": 1,
        "results": fl_round_results,
        "weights": {h: float(w) for h, w in weights.items()},
        "normalized_weights": {h: float(w) for h, w in normalized_weights.items()},
        "total_samples": total_samples
    }
    
    with open("fl_config/example_fairness_weights.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("✓ Example weights saved to: fl_config/example_fairness_weights.json")
    print()


if __name__ == "__main__":
    main()
