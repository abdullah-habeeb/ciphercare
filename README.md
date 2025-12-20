# CipherCare
### Privacy-Preserving Federated Learning Platform for Healthcare AI

CipherCare is a research-oriented, full-stack platform that demonstrates privacy-preserving federated learning for healthcare systems. The platform enables multiple hospitals to collaboratively train machine learning models without sharing raw patient data, addressing privacy, fairness, and compliance challenges in real-world healthcare AI.

The system integrates differential privacy, domain relevance–aware aggregation, and blockchain-inspired auditability to provide a transparent and secure federated learning workflow. This repository emphasizes system architecture, orchestration logic, and reproducible experimentation rather than storing large datasets or trained artifacts.

---

## Key Features

- Federated learning across multiple simulated hospitals  
- Differential privacy for secure gradient updates  
- Domain relevance–weighted aggregation strategies  
- Blockchain-style audit trail for training transparency  
- Hospital-level personalization and evaluation  
- End-to-end experiment orchestration scripts  
- Dashboard-ready backend and frontend integration  

---

## System Architecture

CipherCare is composed of four primary layers:

- **Federated Server**  
  Coordinates training rounds, aggregates client updates, and maintains audit records.

- **Hospital Clients**  
  Train local models on private datasets, apply privacy mechanisms, and never expose raw patient data.

- **Audit Layer**  
  Records immutable metadata for each training round, enabling traceability and compliance analysis.

- **Monitoring Dashboard**  
  Visualizes training progress, performance metrics, and fairness indicators.

---

## Repository Structure

src/                    Core federated learning logic  
fl_utils/               Privacy, personalization, and aggregation utilities  
fl_config/              Configuration files and domain relevance settings  
fl_dashboard/            Monitoring dashboard (frontend + backend)  
examples/               Example aggregation and evaluation scripts  
tests/                  Unit and integration tests  
*.md                    Technical documentation and walkthroughs  


---

## Datasets and Artifacts

Due to size, privacy, and compliance constraints, datasets, trained model weights, and large experiment outputs are intentionally excluded from this repository.
Excluded artifacts include:
- `.npy` datasets
- Trained model weights
- Large evaluation and metric outputs

### Using Your Own Data

1. Prepare or download datasets externally.
2. Place the datasets under:
src/hospital_*/data/

3. Follow the configuration and quickstart guides provided in the documentation files.

This approach keeps the repository lightweight, reproducible, and aligned with industry and academic best practices.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- Virtual environment (recommended)  
- Dependencies listed in `requirements.txt`  

### Installation

```bash
pip install -r requirements.txt

### Running a Federated Simulation

Start the federated server:
python fl_server.py

### Reproducibility Philosophy

CipherCare follows a code-first reproducibility model:
Experiments are recreated via code and configuration files
Results are derived rather than stored
Large artifacts are intentionally excluded from version control
This mirrors real-world ML research and production workflows.

### Use Cases

Privacy-preserving healthcare AI research
Federated learning system prototyping
Academic demonstrations and hackathons
Compliance-aware machine learning system design

### License
ISC License

### Author
Developed as part of an applied research and hackathon initiative focused on secure, distributed machine learning systems.
