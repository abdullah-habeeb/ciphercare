# Automatic Domain Relevance Scoring System - Summary

**Created**: December 12, 2025  
**Status**: ✅ Complete and Tested

---

## 🎯 Overview

Implemented a modular, scalable system to automatically compute domain relevance scores between hospitals in a federated learning network. This system enables fairness-weighted aggregation that respects input modality and disease space overlap.

---

## 📊 Key Features

### 1. Modality Encoding
- Represents each hospital's data sources as weighted vectors
- Computes **cosine similarity** between modality vectors
- **Weight**: 0.7 (70% of final score)

### 2. Label-Space Overlap
- Defines disease classes for each hospital
- Computes **Jaccard similarity** (intersection/union)
- **Weight**: 0.3 (30% of final score)

### 3. Final Score Formula
```
final_score = 0.7 * modality_similarity + 0.3 * label_overlap_score
```

### 4. Fallback Mechanism
- Default score: 0.3 when no overlap exists
- Prevents zero weights in aggregation

### 5. Manual Override Support
- Optional `manual_override_matrix` for expert adjustments
- Useful for known domain relationships

---

## 📁 Files Created

### Configuration
- **`fl_config/hospital_profiles.json`** - Hospital metadata (modalities, labels, demographics)

### Core Module
- **`fl_utils/domain_relevance.py`** - Main computation module (400+ lines)
  - `HospitalConfig` class
  - `cosine_similarity()` function
  - `jaccard_similarity()` function
  - `compute_domain_relevance()` function
  - `compute_domain_relevance_matrix()` function
  - `load_hospital_profiles()` function
  - `save_domain_relevance_matrix()` function
  - `generate_relevance_report()` function

### Outputs
- **`fl_config/domain_relevance_matrix.json`** - Computed relevance scores
- **`fl_config/domain_relevance_report.md`** - Detailed explanation of scores

### Examples & Tests
- **`examples/fairness_weighted_aggregation.py`** - Integration example
- **`tests/test_domain_relevance.py`** - Comprehensive test suite
- **`fl_config/example_fairness_weights.json`** - Example FL weights

---

## 🏥 Hospital Profiles

| Hospital | Modalities | Labels | Samples |
|----------|-----------|--------|---------|
| **A** | ECG | NORM, MI, STTC, CD, HYP | 17,418 |
| **B** | Vitals | Deterioration, Sepsis, Hypoxia, HYP | 800 |
| **C** | CXR | 14 lung pathologies | 160 |
| **D** | ECG | NORM, MI, STTC, CD, HYP | 2,400 |
| **E** | ECG + Vitals + CXR | NORM, MI, STTC, CD, HYP | 2,400 |

---

## 📈 Computed Relevance Scores

### High Relevance (> 0.7)
- **A ↔ D: 1.000** - Both ECG cardiology, identical labels
- **A ↔ E: 0.704** - Shared ECG modality and labels
- **D ↔ E: 0.704** - Shared ECG modality and labels

### Moderate Relevance (0.4 - 0.7)
- **B ↔ E: 0.442** - Shared Vitals modality, minimal label overlap (HYP)
- **C ↔ E: 0.404** - Shared CXR modality, no label overlap

### Low Relevance (< 0.4)
- **A ↔ C: 0.300** - No modality or label overlap (default)
- **B ↔ C: 0.300** - No modality or label overlap (default)
- **C ↔ D: 0.300** - No modality or label overlap (default)
- **A ↔ B: 0.037** - Minimal label overlap (HYP only)
- **B ↔ D: 0.037** - Minimal label overlap (HYP only)

---

## 🔄 Integration with FL Aggregation

### Fairness Weight Formula
```python
weight = 0.6 * AUROC² + 0.3 * (samples / total_samples) + 0.1 * domain_relevance
```

### Example FL Round 1 Weights

| Hospital | AUROC | Samples | Raw Weight | Normalized Weight |
|----------|-------|---------|------------|-------------------|
| **A** | 0.720 | 17,418 | 0.5875 | **26.2%** |
| **B** | 0.960 | 800 | 0.5837 | **26.0%** |
| **E** | 0.750 | 2,400 | 0.4249 | **18.9%** |
| **D** | 0.680 | 2,400 | 0.3595 | **16.0%** |
| **C** | 0.650 | 160 | 0.2882 | **12.8%** |

**Key Insights**:
- Hospital B gets high weight despite small dataset (high AUROC: 0.96)
- Hospital A gets highest weight (large dataset + good AUROC)
- Hospital C gets lowest weight (small dataset + low AUROC)
- Domain relevance provides 10% adjustment based on cross-hospital applicability

---

## 🚀 Usage

### 1. Compute Domain Relevance Matrix
```bash
python fl_utils/domain_relevance.py
```

**Output**:
```
============================================================
Domain Relevance Matrix
============================================================
           A       B       C       D       E
    A   1.00    0.04    0.30    1.00    0.70
    B   0.04    1.00    0.30    0.04    0.44
    C   0.30    0.30    1.00    0.30    0.40
    D   1.00    0.04    0.30    1.00    0.70
    E   0.70    0.44    0.40    0.70    1.00
============================================================
```

### 2. Use in FL Aggregation
```python
from fl_utils.domain_relevance import load_hospital_profiles, compute_domain_relevance_matrix

# Load profiles
hospitals, _ = load_hospital_profiles("fl_config/hospital_profiles.json")

# Compute matrix
matrix = compute_domain_relevance_matrix(hospitals)

# Use in aggregation
relevance_score = matrix[("A", "D")]  # 1.0
```

### 3. Run Example
```bash
python examples/fairness_weighted_aggregation.py
```

---

## ✅ Benefits

### 1. Automatic Scaling
- Works with 5 hospitals or 500 hospitals
- No manual configuration required
- Add new hospitals by updating `hospital_profiles.json`

### 2. Balanced Influence
- Prevents irrelevant hospitals from dominating aggregation
- Rewards cross-domain applicability
- Maintains fairness across modalities

### 3. Cross-Modality Awareness
- ECG hospitals collaborate more with other ECG hospitals
- Multimodal hospitals (E) bridge different domains
- Prevents "noise" from unrelated specialties

### 4. Easy Maintenance
- Single JSON file for all hospital metadata
- Automatic recomputation when profiles change
- Detailed reports for auditing

### 5. Transparency
- Clear formula: 0.7*modality + 0.3*labels
- Detailed report shows breakdown for each pair
- JSON output for programmatic access

---

## 🧪 Testing

### Test Suite: `tests/test_domain_relevance.py`

**Test Coverage**:
- ✅ Cosine similarity (identical, orthogonal, opposite, zero vectors)
- ✅ Jaccard similarity (identical, disjoint, partial overlap, empty sets)
- ✅ Domain relevance (same modality/labels, different modality, no overlap)
- ✅ Matrix symmetry
- ✅ Self-relevance (always 1.0)
- ✅ Manual overrides
- ✅ Real hospital profiles (A-D high, A-B low, E multimodal)

**Run Tests**:
```bash
python -m pytest tests/test_domain_relevance.py -v
```

---

## 📊 Example Output Files

### `domain_relevance_matrix.json`
```json
{
  "A-D": 1.0,
  "A-E": 0.704,
  "B-E": 0.442,
  "A-B": 0.037
}
```

### `domain_relevance_report.md`
```markdown
### A <-> D: **1.000**
- Modality Similarity: 1.000
- Label Overlap: 1.000
- Shared Labels: CD, HYP, MI, NORM, STTC
```

### `example_fairness_weights.json`
```json
{
  "round": 1,
  "normalized_weights": {
    "A": 0.262,
    "B": 0.260,
    "E": 0.189,
    "D": 0.160,
    "C": 0.128
  }
}
```

---

## 🔮 Future Enhancements

### 1. Dynamic Relevance
- Update relevance scores based on FL performance
- Learn optimal weights from historical data

### 2. Demographic Weighting
- Add age/gender distribution similarity
- Weight by demographic overlap

### 3. Performance-Based Adjustment
- Boost relevance for hospitals that improve each other
- Penalize negative transfer

### 4. Temporal Relevance
- Account for data recency
- Weight recent data more heavily

---

## 📚 API Reference

### `compute_domain_relevance_matrix()`
```python
def compute_domain_relevance_matrix(
    hospitals: List[HospitalConfig],
    modality_order: List[str] = None,
    modality_weight: float = 0.7,
    label_weight: float = 0.3,
    default_score: float = 0.3,
    manual_overrides: Dict[str, float] = None
) -> Dict[Tuple[str, str], float]
```

**Returns**: Symmetric matrix mapping `(hospital_i, hospital_j)` to relevance score in [0, 1]

---

## ✨ Summary

**Status**: ✅ Fully Implemented and Tested

**Files Created**: 7
- 1 configuration file
- 1 core module (400+ lines)
- 2 output files (matrix + report)
- 1 example script
- 1 test suite
- 1 example output

**Key Metrics**:
- **Computation Time**: < 1 second for 5 hospitals
- **Matrix Size**: 25 entries (5x5 symmetric)
- **Accuracy**: Validated against expected scores
- **Scalability**: O(n²) for n hospitals

**Integration Ready**: ✅ Can be used immediately in FL server aggregation

---

**Generated**: December 12, 2025 @ 17:30 IST
