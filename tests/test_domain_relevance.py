"""
Tests for domain relevance scoring system
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append('.')
from fl_utils.domain_relevance import (
    HospitalConfig,
    cosine_similarity,
    jaccard_similarity,
    compute_domain_relevance,
    compute_domain_relevance_matrix,
    load_hospital_profiles
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""
    
    def test_identical_vectors(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    
    def test_opposite_vectors(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([-1, 0, 0])
        # Using abs() so this should be 1.0
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
    
    def test_zero_vector(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 0, 0])
        assert cosine_similarity(vec1, vec2) == 0.0


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""
    
    def test_identical_sets(self):
        set1 = {"A", "B", "C"}
        set2 = {"A", "B", "C"}
        assert jaccard_similarity(set1, set2) == 1.0
    
    def test_disjoint_sets(self):
        set1 = {"A", "B"}
        set2 = {"C", "D"}
        assert jaccard_similarity(set1, set2) == 0.0
    
    def test_partial_overlap(self):
        set1 = {"A", "B", "C"}
        set2 = {"B", "C", "D"}
        # Intersection: {B, C} = 2, Union: {A, B, C, D} = 4
        assert jaccard_similarity(set1, set2) == pytest.approx(0.5)
    
    def test_empty_sets(self):
        set1 = set()
        set2 = set()
        assert jaccard_similarity(set1, set2) == 1.0


class TestDomainRelevance:
    """Test domain relevance computation."""
    
    def test_identical_hospitals(self):
        """Two hospitals with same modality and labels should have score 1.0"""
        h1 = HospitalConfig({
            "id": "A",
            "name": "Hospital A",
            "modalities": {"ECG": 1.0, "Vitals": 0.0},
            "labels": ["MI", "HYP"]
        })
        h2 = HospitalConfig({
            "id": "B",
            "name": "Hospital B",
            "modalities": {"ECG": 1.0, "Vitals": 0.0},
            "labels": ["MI", "HYP"]
        })
        
        score = compute_domain_relevance(h1, h2, ["ECG", "Vitals"])
        assert score == pytest.approx(1.0)
    
    def test_different_modalities_same_labels(self):
        """Different modalities but same labels"""
        h1 = HospitalConfig({
            "id": "A",
            "name": "Hospital A",
            "modalities": {"ECG": 1.0, "Vitals": 0.0},
            "labels": ["MI", "HYP"]
        })
        h2 = HospitalConfig({
            "id": "B",
            "name": "Hospital B",
            "modalities": {"ECG": 0.0, "Vitals": 1.0},
            "labels": ["MI", "HYP"]
        })
        
        score = compute_domain_relevance(h1, h2, ["ECG", "Vitals"])
        # Modality sim = 0.0, Label sim = 1.0
        # Score = 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        assert score == pytest.approx(0.3)
    
    def test_no_overlap(self):
        """No modality or label overlap should use default score"""
        h1 = HospitalConfig({
            "id": "A",
            "name": "Hospital A",
            "modalities": {"ECG": 1.0, "Vitals": 0.0},
            "labels": ["MI"]
        })
        h2 = HospitalConfig({
            "id": "B",
            "name": "Hospital B",
            "modalities": {"ECG": 0.0, "Vitals": 1.0},
            "labels": ["Sepsis"]
        })
        
        score = compute_domain_relevance(h1, h2, ["ECG", "Vitals"], default_score=0.3)
        # Modality sim = 0.0, Label sim = 0.0
        # Should use default
        assert score == pytest.approx(0.3)


class TestDomainRelevanceMatrix:
    """Test full matrix computation."""
    
    def test_matrix_symmetry(self):
        """Matrix should be symmetric"""
        hospitals = [
            HospitalConfig({
                "id": "A",
                "name": "Hospital A",
                "modalities": {"ECG": 1.0},
                "labels": ["MI"]
            }),
            HospitalConfig({
                "id": "B",
                "name": "Hospital B",
                "modalities": {"Vitals": 1.0},
                "labels": ["Sepsis"]
            })
        ]
        
        matrix = compute_domain_relevance_matrix(hospitals)
        
        assert matrix[("A", "B")] == matrix[("B", "A")]
    
    def test_self_relevance(self):
        """Self-relevance should always be 1.0"""
        hospitals = [
            HospitalConfig({
                "id": "A",
                "name": "Hospital A",
                "modalities": {"ECG": 1.0},
                "labels": ["MI"]
            })
        ]
        
        matrix = compute_domain_relevance_matrix(hospitals)
        
        assert matrix[("A", "A")] == 1.0
    
    def test_manual_override(self):
        """Manual overrides should take precedence"""
        hospitals = [
            HospitalConfig({
                "id": "A",
                "name": "Hospital A",
                "modalities": {"ECG": 1.0},
                "labels": ["MI"]
            }),
            HospitalConfig({
                "id": "B",
                "name": "Hospital B",
                "modalities": {"Vitals": 1.0},
                "labels": ["Sepsis"]
            })
        ]
        
        manual_overrides = {"A-B": 0.95}
        matrix = compute_domain_relevance_matrix(hospitals, manual_overrides=manual_overrides)
        
        assert matrix[("A", "B")] == 0.95
        assert matrix[("B", "A")] == 0.95


class TestRealHospitalProfiles:
    """Test with actual hospital profiles."""
    
    def test_load_profiles(self):
        """Test loading hospital profiles from JSON"""
        hospitals, overrides = load_hospital_profiles("fl_config/hospital_profiles.json")
        
        assert len(hospitals) == 5
        assert hospitals[0].id == "A"
        assert hospitals[1].id == "B"
    
    def test_hospital_a_d_high_relevance(self):
        """Hospitals A and D (both ECG cardiology) should have high relevance"""
        hospitals, _ = load_hospital_profiles("fl_config/hospital_profiles.json")
        matrix = compute_domain_relevance_matrix(hospitals)
        
        # A and D are both ECG with same labels
        assert matrix[("A", "D")] > 0.9
    
    def test_hospital_a_b_low_relevance(self):
        """Hospitals A (ECG) and B (Vitals) should have low relevance"""
        hospitals, _ = load_hospital_profiles("fl_config/hospital_profiles.json")
        matrix = compute_domain_relevance_matrix(hospitals)
        
        # A is ECG, B is Vitals, minimal label overlap
        assert matrix[("A", "B")] < 0.2
    
    def test_hospital_e_multimodal_high_relevance(self):
        """Hospital E (multimodal) should have high relevance with A, D"""
        hospitals, _ = load_hospital_profiles("fl_config/hospital_profiles.json")
        matrix = compute_domain_relevance_matrix(hospitals)
        
        # E has ECG + Vitals + CXR, shares ECG and labels with A
        assert matrix[("A", "E")] > 0.6
        assert matrix[("D", "E")] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
