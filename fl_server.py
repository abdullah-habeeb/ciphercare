"""
Federated Learning Server for Multi-Hospital Collaboration
Coordinates training between Hospital A, Hospital D, and future nodes.
"""
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    """
    # Extract accuracies/AUROCs
    aurocs = [num_examples * m["auroc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate
    aggregated_auroc = sum(aurocs) / sum(examples) if sum(examples) > 0 else 0.0
    
    return {"auroc": aggregated_auroc}

def main():
    """
    Start the Flower federated learning server.
    
    Configuration:
    - Strategy: FedAvg (Federated Averaging)
    - Min clients: 2 (Hospital A + Hospital D)
    - Rounds: 5
    - Fraction fit: 1.0 (all available clients)
    """
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=2,  # Minimum 2 clients (A + D)
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    print("="*60)
    print("Federated Learning Server Starting")
    print("="*60)
    print("Strategy: FedAvg")
    print("Min clients: 2 (Hospital A + Hospital D)")
    print("Server address: 0.0.0.0:8080")
    print("\nWaiting for clients to connect...")
    print("="*60)
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
