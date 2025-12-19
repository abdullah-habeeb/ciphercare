# Domain Relevance Report

## Hospital Profiles

### Hospital A: General Cardiology
- **Specialty**: Cardiology
- **Demographic**: General (18-85 yrs)
- **Samples**: 17,418
- **Modalities**: ECG
- **Labels**: CD, HYP, MI, NORM, STTC

### Hospital B: ICU Deterioration Monitoring
- **Specialty**: Pulmonary/General Monitoring
- **Demographic**: ICU/Trauma
- **Samples**: 800
- **Modalities**: Vitals
- **Labels**: Deterioration, HYP, Hypoxia, Sepsis

### Hospital C: Respiratory Diagnostics
- **Specialty**: Respiratory Diagnostics
- **Demographic**: Pediatric & Adult
- **Samples**: 160
- **Modalities**: CXR
- **Labels**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

### Hospital D: Geriatric Cardiology
- **Specialty**: Elder Cardiology
- **Demographic**: Geriatric (60+)
- **Samples**: 2,400
- **Modalities**: ECG
- **Labels**: CD, HYP, MI, NORM, STTC

### Hospital E: Multimodal Fusion
- **Specialty**: Multimodal Fusion (Cross-Domain)
- **Demographic**: General
- **Samples**: 2,400
- **Modalities**: ECG, Vitals, CXR
- **Labels**: CD, HYP, MI, NORM, STTC


## Pairwise Relevance Scores

### A <-> D: **1.000**
- Modality Similarity: 1.000
- Label Overlap: 1.000
- Shared Labels: CD, HYP, MI, NORM, STTC

### A <-> E: **0.704**
- Modality Similarity: 0.577
- Label Overlap: 1.000
- Shared Labels: CD, HYP, MI, NORM, STTC

### D <-> E: **0.704**
- Modality Similarity: 0.577
- Label Overlap: 1.000
- Shared Labels: CD, HYP, MI, NORM, STTC

### B <-> E: **0.442**
- Modality Similarity: 0.577
- Label Overlap: 0.125
- Shared Labels: HYP

### C <-> E: **0.404**
- Modality Similarity: 0.577
- Label Overlap: 0.000
- Shared Labels: None

### A <-> C: **0.300**
- Modality Similarity: 0.000
- Label Overlap: 0.000
- Shared Labels: None

### B <-> C: **0.300**
- Modality Similarity: 0.000
- Label Overlap: 0.000
- Shared Labels: None

### C <-> D: **0.300**
- Modality Similarity: 0.000
- Label Overlap: 0.000
- Shared Labels: None

### A <-> B: **0.037**
- Modality Similarity: 0.000
- Label Overlap: 0.125
- Shared Labels: HYP

### B <-> D: **0.037**
- Modality Similarity: 0.000
- Label Overlap: 0.125
- Shared Labels: HYP
