# CipherCare

### Privacy-Preserving Federated Learning for Healthcare AI

CipherCare is a federated learning platform that simulates five hospitals collaboratively training diagnostic models — ECG classification, ICU deterioration risk, chest X-ray screening, and a multimodal fusion task — without ever sharing raw patient data. Each hospital trains locally; only model updates, protected by differential privacy, leave the client. A fairness-aware aggregation strategy and a hash-chained audit trail sit on top of a standard federated learning loop.

Originally built for a hackathon, the system now runs end-to-end on real public medical datasets and has been verified with live training runs — not just described in documentation.

---

## What it actually does

- **Federated training across 5 simulated hospitals**, coordinated by a custom [Flower](https://flower.ai/) strategy (`FedProxFairness`) combining [FedProx](https://arxiv.org/abs/1812.06127) with a fairness-weighted aggregation rule:

  ```
  weight = 0.6 · AUROC² + 0.3 · (samples / total_samples) + 0.1 · domain_relevance
  ```

  so a hospital with a smaller but higher-quality dataset isn't drowned out by raw sample count alone.

- **Automatic domain-relevance scoring** between hospitals, computed from modality overlap (cosine similarity) and disease-label overlap (Jaccard similarity) — no manual configuration of which hospitals "trust" each other.

- **Differential privacy** on every client update: gradient clipping + calibrated Gaussian noise, with per-hospital noise scaled to sample count so every participant reaches the same privacy budget (ε, δ).

- **A hash-chained audit log** for every round, DP guarantee, and model update — tamper-evident, not a deployed blockchain (see [Known Limitations](#known-limitations)).

- **Real local-only baselines and post-FL personalization**, so the system reports what each hospital could do alone, after federation, and after fine-tuning its own head on the shared model — not simulated numbers.

---

## Architecture

| Hospital | Task | Real data source | Samples |
|---|---|---|---|
| A | General cardiology (ECG) | [PTB-XL](https://physionet.org/content/ptb-xl/) | 17,418 |
| B | ICU deterioration risk | Structured synthetic vitals (rule-based, not random) | 1,000 |
| C | Chest X-ray screening | [NIH ChestX-ray14](https://huggingface.co/datasets/Sohaibsoussi/NIH-Chest-X-ray-dataset-small) | 4,326 |
| D | Geriatric cardiology (age 60+) | PTB-XL, age-filtered | 12,049 |
| E | Multimodal fusion | Combined ECG + vitals + X-ray features | 800 |

All five hospitals share one compact model architecture (`UnifiedFLModel`, a 3-layer MLP) so their updates can be meaningfully averaged despite very different raw inputs — see [Known Limitations](#known-limitations) for the trade-off this involves.

```
fl_server_enhanced.py          FL server: FedProx + fairness aggregation + DP + audit logging
run_hospital_{a,b,c,d,e}_client_enhanced.py   Hospital clients
run_local_baselines.py         Real local-only (non-federated) baseline per hospital
run_personalization.py         Real fine-tuning of the aggregated model per hospital
fl_utils/                      DP, domain relevance, blockchain audit, personalization utilities
fl_config/                     Hospital profiles and domain relevance config
fl_dashboard/                  Monitoring dashboard (FastAPI backend + React frontend)
frontend_integration/          Secondary product-style dashboard (IoMT monitoring, hospital views)
tests/                         Unit tests
```

---

## Verified results

The numbers below are from an actual live 3-round federated run on real data (not simulated or hardcoded), with the blockchain audit chain intact and cryptographically verified:

| Hospital | AUROC |
|---|---|
| A (ECG) | 0.695 |
| B (vitals) | 0.912 |
| C (chest X-ray) | 0.622 |
| D (geriatric ECG) | 0.668 |
| E (fusion) | 0.781 |
| **Average** | **0.705** |

Average AUROC across rounds: 0.583 → 0.644 → 0.705.

---

## Getting started

### Prerequisites

- Python 3.11+
- ~5GB free disk space for the real PTB-XL and chest X-ray datasets

### Installation

```bash
pip install -r requirements.txt
```

### Preparing real data

Datasets are not committed to this repository (see below). To reproduce real results:

```bash
# ECG (Hospitals A & D) — requires downloading PTB-XL from PhysioNet first
python src/hospital_a/utils/process_raw.py --db-path <path>/ptbxl_database.csv --records-dir <path>
python src/hospital_d/utils/extract_geriatric.py --db-path <path>/ptbxl_database.csv --records-dir <path>

# Vitals (Hospital B) — generates structured synthetic data, no download needed
python src/hospital_b/preprocess_data.py

# Chest X-rays (Hospital C) — streams real images from HuggingFace
python src/hospital_c/download_data.py

# Fusion (Hospital E) — combines the above once A, B, and C are ready
python src/hospital_e/utils/build_fusion_data.py
```

### Running federated training

```bash
python fl_server_enhanced.py            # start the FL server
python run_hospital_a_client_enhanced.py  # in separate terminals, one per hospital
python run_hospital_b_client_enhanced.py
python run_hospital_c_client_enhanced.py
python run_hospital_d_client_enhanced.py
python run_hospital_e_client_enhanced.py
```

Results land in `fl_results/`: per-round aggregation logs, the audit chain, and (after each round) a saved global model checkpoint.

### Local baselines and personalization

```bash
python run_local_baselines.py     # real non-federated baseline per hospital
python run_personalization.py     # real fine-tuning of the federated model per hospital
```

---

## Known limitations

Documented here rather than left for someone else to discover:

- **Shared model architecture.** All five hospitals train the same lightweight MLP rather than per-modality architectures (a real CNN for X-rays, a sequence model for ECG). This keeps the system's federated averaging simple and fast to reproduce, at the cost of ceiling model quality — a production system would use modality-appropriate encoders.
- **Hospital B's data is synthetic.** It's rule-based (genuine, learnable structure) rather than random, but not real clinical vitals.
- **Hospital E's fusion dataset combines unrelated datasets.** PTB-XL, the synthetic vitals, and the NIH X-ray images have no real linked patient identity; the fusion set pairs them for architecture demonstration purposes, not as a genuine multimodal clinical cohort.
- **The audit trail is a hash-chained log, not a deployed blockchain** — no consensus, no decentralization. It's tamper-evident and exportable to a real chain, but isn't one today.
- **This repository does not include the raw datasets or trained weights.** They're excluded by `.gitignore` (size, licensing) — use the data-preparation scripts above to regenerate them locally.

---

## Tech stack

Python · PyTorch · [Flower](https://flower.ai/) · scikit-learn · FastAPI · SQLAlchemy · React · Vite · [PTB-XL](https://physionet.org/content/ptb-xl/) · [NIH ChestX-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

## License

ISC

## Acknowledgments

- Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* PhysioNet, 2020.
- NIH Clinical Center. *ChestX-ray14* dataset.
