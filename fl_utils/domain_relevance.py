"""
Automatic Domain Relevance Scoring for Federated Learning

Computes domain relevance scores between hospitals based on:
1. Modality similarity (cosine similarity of modality vectors)
2. Label-space overlap (Jaccard similarity of disease labels)

Final score = 0.7 * modality_similarity + 0.3 * label_overlap_score
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Set
from pathlib import Path


class HospitalConfig:
    """Configuration for a single hospital in the FL network."""
    
    def __init__(self, config_dict: Dict):
        self.id = config_dict["id"]
        self.name = config_dict["name"]
        self.modalities = config_dict["modalities"]  # Dict[str, float]
        self.labels = set(config_dict["labels"])  # Set[str]
        self.specialty = config_dict.get("specialty", "Unknown")
        self.demographic = config_dict.get("demographic", "Unknown")
        self.num_samples = config_dict.get("num_samples", 0)
    
    def get_modality_vector(self, modality_order: List[str]) -> np.ndarray:
        """
        Convert modality dict to vector following specified order.
        
        Args:
            modality_order: List of modality names (e.g., ['ECG', 'Vitals', 'CXR', 'Audio'])
        
        Returns:
            numpy array of modality weights
        """
        return np.array([self.modalities.get(m, 0.0) for m in modality_order])
    
    def __repr__(self):
        return f"Hospital({self.id}: {self.name})"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2: numpy arrays
    
    Returns:
        Cosine similarity in [0, 1] (using absolute value to ensure positive)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return abs(similarity)  # Ensure positive


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two label sets.
    
    Jaccard = |intersection| / |union|
    
    Args:
        set1, set2: Sets of disease labels
    
    Returns:
        Jaccard similarity in [0, 1]
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0  # Both empty = perfect match
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def compute_domain_relevance(
    hospital_i: HospitalConfig,
    hospital_j: HospitalConfig,
    modality_order: List[str],
    modality_weight: float = 0.7,
    label_weight: float = 0.3,
    default_score: float = 0.3
) -> float:
    """
    Compute domain relevance score between two hospitals.
    
    Args:
        hospital_i, hospital_j: Hospital configurations
        modality_order: Ordered list of modality names
        modality_weight: Weight for modality similarity (default: 0.7)
        label_weight: Weight for label overlap (default: 0.3)
        default_score: Fallback score if no overlap (default: 0.3)
    
    Returns:
        Domain relevance score in [0, 1]
    """
    # 1. Modality similarity
    vec_i = hospital_i.get_modality_vector(modality_order)
    vec_j = hospital_j.get_modality_vector(modality_order)
    modality_sim = cosine_similarity(vec_i, vec_j)
    
    # 2. Label overlap
    label_sim = jaccard_similarity(hospital_i.labels, hospital_j.labels)
    
    # 3. Combined score
    final_score = modality_weight * modality_sim + label_weight * label_sim
    
    # 4. Apply default if no meaningful overlap
    if modality_sim == 0.0 and label_sim == 0.0:
        final_score = default_score
    
    return final_score


def compute_domain_relevance_matrix(
    hospitals: List[HospitalConfig],
    modality_order: List[str] = None,
    modality_weight: float = 0.7,
    label_weight: float = 0.3,
    default_score: float = 0.3,
    manual_overrides: Dict[str, float] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute full domain relevance matrix for all hospital pairs.
    
    Args:
        hospitals: List of HospitalConfig objects
        modality_order: Ordered list of modality names (auto-detected if None)
        modality_weight: Weight for modality similarity (default: 0.7)
        label_weight: Weight for label overlap (default: 0.3)
        default_score: Fallback score if no overlap (default: 0.3)
        manual_overrides: Dict mapping "Hospital_i-Hospital_j" to manual score
    
    Returns:
        Dict mapping (hospital_i_id, hospital_j_id) to relevance score
        Matrix is symmetric: score(i,j) = score(j,i)
    """
    # Auto-detect modality order if not provided
    if modality_order is None:
        all_modalities = set()
        for h in hospitals:
            all_modalities.update(h.modalities.keys())
        modality_order = sorted(all_modalities)
    
    # Initialize matrix
    matrix = {}
    
    # Compute pairwise scores
    for i, hospital_i in enumerate(hospitals):
        for j, hospital_j in enumerate(hospitals):
            if i == j:
                # Self-relevance is always 1.0
                matrix[(hospital_i.id, hospital_j.id)] = 1.0
            elif i < j:
                # Compute score for (i, j)
                score = compute_domain_relevance(
                    hospital_i, hospital_j,
                    modality_order,
                    modality_weight,
                    label_weight,
                    default_score
                )
                
                # Check for manual override
                if manual_overrides:
                    key1 = f"{hospital_i.id}-{hospital_j.id}"
                    key2 = f"{hospital_j.id}-{hospital_i.id}"
                    if key1 in manual_overrides:
                        score = manual_overrides[key1]
                    elif key2 in manual_overrides:
                        score = manual_overrides[key2]
                
                # Store symmetric entries
                matrix[(hospital_i.id, hospital_j.id)] = score
                matrix[(hospital_j.id, hospital_i.id)] = score
    
    return matrix


def load_hospital_profiles(config_path: str) -> Tuple[List[HospitalConfig], Dict[str, float]]:
    """
    Load hospital profiles from JSON configuration file.
    
    Args:
        config_path: Path to hospital_profiles.json
    
    Returns:
        Tuple of (list of HospitalConfig objects, manual_overrides dict)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    hospitals = [HospitalConfig(h) for h in config["hospitals"]]
    manual_overrides = config.get("manual_overrides", {})
    
    # Filter out comment keys
    manual_overrides = {k: v for k, v in manual_overrides.items() if not k.startswith("_")}
    
    return hospitals, manual_overrides


def save_domain_relevance_matrix(
    matrix: Dict[Tuple[str, str], float],
    output_path: str
):
    """
    Save domain relevance matrix to JSON file.
    
    Args:
        matrix: Domain relevance matrix
        output_path: Path to save JSON file
    """
    # Convert tuple keys to strings for JSON serialization
    serializable_matrix = {
        f"{i}-{j}": score for (i, j), score in matrix.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_matrix, f, indent=2)
    
    print(f"+ Domain relevance matrix saved to: {output_path}")


def print_domain_relevance_matrix(
    matrix: Dict[Tuple[str, str], float],
    hospitals: List[HospitalConfig]
):
    """
    Pretty-print domain relevance matrix as a table.
    
    Args:
        matrix: Domain relevance matrix
        hospitals: List of HospitalConfig objects
    """
    hospital_ids = [h.id for h in hospitals]
    
    print("\n" + "="*60)
    print("Domain Relevance Matrix")
    print("="*60)
    
    # Header
    print(f"{'':>5}", end="")
    for h_id in hospital_ids:
        print(f"{h_id:>8}", end="")
    print()
    
    # Rows
    for i, h_i in enumerate(hospital_ids):
        print(f"{h_i:>5}", end="")
        for j, h_j in enumerate(hospital_ids):
            score = matrix.get((h_i, h_j), 0.0)
            print(f"{score:>8.2f}", end="")
        print()
    
    print("="*60)


def generate_relevance_report(
    matrix: Dict[Tuple[str, str], float],
    hospitals: List[HospitalConfig],
    output_path: str = None
):
    """
    Generate a detailed report explaining relevance scores.
    
    Args:
        matrix: Domain relevance matrix
        hospitals: List of HospitalConfig objects
        output_path: Optional path to save report (prints to console if None)
    """
    report_lines = []
    report_lines.append("# Domain Relevance Report\n")
    report_lines.append("## Hospital Profiles\n")
    
    for h in hospitals:
        report_lines.append(f"### Hospital {h.id}: {h.name}")
        report_lines.append(f"- **Specialty**: {h.specialty}")
        report_lines.append(f"- **Demographic**: {h.demographic}")
        report_lines.append(f"- **Samples**: {h.num_samples:,}")
        report_lines.append(f"- **Modalities**: {', '.join([k for k, v in h.modalities.items() if v > 0])}")
        report_lines.append(f"- **Labels**: {', '.join(sorted(h.labels))}")
        report_lines.append("")
    
    report_lines.append("\n## Pairwise Relevance Scores\n")
    
    # Sort pairs by relevance score (descending)
    pairs = [(i, j, score) for (i, j), score in matrix.items() if i < j]
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, score in pairs:
        h_i = next(h for h in hospitals if h.id == i)
        h_j = next(h for h in hospitals if h.id == j)
        
        # Compute individual components
        modality_order = sorted(set(h_i.modalities.keys()).union(h_j.modalities.keys()))
        vec_i = h_i.get_modality_vector(modality_order)
        vec_j = h_j.get_modality_vector(modality_order)
        modality_sim = cosine_similarity(vec_i, vec_j)
        label_sim = jaccard_similarity(h_i.labels, h_j.labels)
        
        report_lines.append(f"### {i} <-> {j}: **{score:.3f}**")
        report_lines.append(f"- Modality Similarity: {modality_sim:.3f}")
        report_lines.append(f"- Label Overlap: {label_sim:.3f}")
        report_lines.append(f"- Shared Labels: {', '.join(sorted(h_i.labels.intersection(h_j.labels))) or 'None'}")
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"+ Relevance report saved to: {output_path}")
    else:
        print(report)


def main():
    """
    Main function to compute and save domain relevance matrix.
    """
    # Paths
    config_path = "fl_config/hospital_profiles.json"
    matrix_output_path = "fl_config/domain_relevance_matrix.json"
    report_output_path = "fl_config/domain_relevance_report.md"
    
    print("="*60)
    print("Automatic Domain Relevance Scoring")
    print("="*60)
    
    # Load hospital profiles
    print(f"\n1. Loading hospital profiles from: {config_path}")
    hospitals, manual_overrides = load_hospital_profiles(config_path)
    print(f"   + Loaded {len(hospitals)} hospitals")
    
    # Compute matrix
    print(f"\n2. Computing domain relevance matrix...")
    print(f"   - Modality weight: 0.7")
    print(f"   - Label weight: 0.3")
    print(f"   - Default score: 0.3")
    
    matrix = compute_domain_relevance_matrix(
        hospitals,
        modality_weight=0.7,
        label_weight=0.3,
        default_score=0.3,
        manual_overrides=manual_overrides
    )
    print(f"   + Computed {len(matrix)} pairwise scores")
    
    # Print matrix
    print_domain_relevance_matrix(matrix, hospitals)
    
    # Save matrix
    print(f"\n3. Saving results...")
    save_domain_relevance_matrix(matrix, matrix_output_path)
    
    # Generate report
    generate_relevance_report(matrix, hospitals, report_output_path)
    
    print("\n" + "="*60)
    print("+ Domain relevance scoring complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Matrix: {matrix_output_path}")
    print(f"  - Report: {report_output_path}")


if __name__ == "__main__":
    main()
