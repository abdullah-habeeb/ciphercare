"""
Federated Learning Configuration

Centralized configuration for FL server and clients.
"""

# FL Server Configuration
FL_CONFIG = {
    # Server settings
    "server_address": "0.0.0.0:8080",
    "num_rounds": 3,
    "min_fit_clients": 5,
    "min_evaluate_clients": 5,
    "min_available_clients": 5,
    "fraction_fit": 1.0,  # Use all available clients
    "fraction_evaluate": 1.0,
    
    # FedProx settings
    "fedprox_mu": 0.01,  # Proximal term coefficient
    
    # Differential Privacy settings
    "dp_epsilon": 5.0,  # Privacy budget
    "dp_delta": 1e-5,  # Failure probability
    "max_grad_norm": 1.0,  # Gradient clipping threshold
    
    # Fairness-weighted aggregation
    "fairness_weights": {
        "auroc": 0.6,  # Weight for AUROC component
        "samples": 0.3,  # Weight for sample count component
        "domain_relevance": 0.1  # Weight for domain relevance component
    },
    
    # Domain relevance settings
    "domain_relevance_config": "fl_config/hospital_profiles.json",
    "modality_weight": 0.7,  # Weight for modality similarity
    "label_weight": 0.3,  # Weight for label overlap
    "default_relevance": 0.3,  # Default score when no overlap
    
    # Logging settings
    "log_dir": "fl_results",
    "save_checkpoints": True,
    "checkpoint_dir": "fl_results/checkpoints",
    "save_metrics": True,
    "metrics_dir": "fl_results/metrics"
}

# Client-specific configurations
CLIENT_CONFIGS = {
    "A": {
        "hospital_id": "A",
        "hospital_name": "General Cardiology",
        "data_path": "src/hospital_a/data",
        "model_type": "ECGClassifier",
        "num_samples": 17418,
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 1e-4
    },
    "B": {
        "hospital_id": "B",
        "hospital_name": "ICU Deterioration",
        "data_path": "data/hospital_b",
        "model_type": "MLP",
        "num_samples": 800,
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 1e-3
    },
    "C": {
        "hospital_id": "C",
        "hospital_name": "Respiratory Diagnostics",
        "data_path": "data/hospital_c",
        "model_type": "ResNet50",
        "num_samples": 160,
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 1e-4
    },
    "D": {
        "hospital_id": "D",
        "hospital_name": "Geriatric Cardiology",
        "data_path": "src/hospital_d/data",
        "model_type": "ECGClassifier",
        "num_samples": 2400,
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 1e-4
    },
    "E": {
        "hospital_id": "E",
        "hospital_name": "Multimodal Fusion",
        "data_path": "data/hospital_e",
        "model_type": "FusionClassifier",
        "num_samples": 2400,
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 1e-4
    }
}


def get_fl_config():
    """Get FL configuration."""
    return FL_CONFIG


def get_client_config(hospital_id: str):
    """Get configuration for specific hospital client."""
    return CLIENT_CONFIGS.get(hospital_id)


def get_all_client_configs():
    """Get all client configurations."""
    return CLIENT_CONFIGS
