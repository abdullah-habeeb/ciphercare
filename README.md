CipherCare
Federated Learning Platform for Privacy-Preserving Healthcare AI
Overview

CipherCare is a research-oriented, full-stack platform that demonstrates privacy-preserving federated learning in healthcare systems, enhanced with differential privacy, domain relevance weighting, and blockchain-based auditability.

The system simulates multiple hospitals collaboratively training machine learning models without sharing raw patient data, while maintaining transparency, fairness, and traceability across training rounds.

This repository focuses on system design, orchestration logic, and reproducible experimentation, rather than serving as a data dump.

Key Features

Federated learning across multiple simulated hospitals

Differential privacy mechanisms for gradient updates

Domain relevance–aware aggregation strategies

Blockchain-style audit trail for model updates

Hospital-level personalization and evaluation

Dashboard-ready backend and frontend integration

End-to-end experiment orchestration scripts

Architecture Overview

Federated Server

Coordinates training rounds

Aggregates client updates

Maintains audit records

Hospital Clients

Train local models on private data

Apply privacy and personalization logic

Audit Layer

Records immutable training metadata

Enables traceability and compliance reasoning

Dashboard

Visualizes training progress, metrics, and fairness indicators

Repository Structure
src/                    Core federated learning logic
fl_utils/               Privacy, personalization, and aggregation utilities
fl_config/              Configuration files and domain relevance settings
fl_dashboard/            Monitoring dashboard (frontend + backend)
examples/               Example aggregation and evaluation scripts
tests/                  Unit and integration tests
*.md                    Technical documentation and walkthroughs

Dataset & Artifacts (Important)

Due to size and privacy constraints, datasets, trained models, and experiment outputs are not included in this repository.

These artifacts include:

.npy datasets

trained model weights

large evaluation outputs

How to use your own data

Download or prepare datasets externally

Place them under:

src/hospital_*/data/


Follow the configuration guides provided in the documentation files

This approach keeps the repository lightweight, compliant, and reproducible.

Getting Started
Prerequisites

Python 3.9+

Virtual environment recommended

Common ML libraries (see requirements.txt)

Setup
pip install -r requirements.txt

Run a federated simulation
python fl_server.py


Run individual hospital clients in separate terminals as documented in the quickstart guides.

Reproducibility Philosophy

CipherCare is designed around reproducible systems, not static artifacts.

Experiments are recreated via code

Results are derived, not stored

Configurations are explicit and versioned

This mirrors real-world ML research and production workflows.

Use Cases

Privacy-preserving healthcare AI research

Federated learning system prototyping

Academic demonstrations and hackathons

Compliance-aware ML system design

License

ISC License

Author

Developed as part of an applied research and hackathon initiative focused on secure, distributed machine learning systems.
