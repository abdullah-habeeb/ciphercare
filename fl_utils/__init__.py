"""
Federated Learning Utilities

Modules:
- domain_relevance: Automatic domain relevance scoring for FL aggregation
"""

from .domain_relevance import (
    HospitalConfig,
    compute_domain_relevance_matrix,
    load_hospital_profiles,
    save_domain_relevance_matrix,
    print_domain_relevance_matrix,
    generate_relevance_report
)

__all__ = [
    'HospitalConfig',
    'compute_domain_relevance_matrix',
    'load_hospital_profiles',
    'save_domain_relevance_matrix',
    'print_domain_relevance_matrix',
    'generate_relevance_report'
]
