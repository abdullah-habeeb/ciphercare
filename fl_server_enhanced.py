"""
Enhanced Federated Learning Server with FedProx + Fairness Weighting + DP

Features:
- FedProx strategy with proximal term (µ=0.01)
- Fairness-weighted aggregation (0.6*AUROC² + 0.3*samples + 0.1*domain_relevance)
- Differential privacy support (epsilon≤5)
- Automatic domain relevance scoring
- Comprehensive logging and metrics tracking
"""

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import Strategy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

# Import our custom modules
import sys
sys.path.append('.')
from fl_config import get_fl_config
from fl_utils.domain_relevance import load_hospital_profiles, compute_domain_relevance_matrix
from fl_utils.blockchain_audit import BlockchainAuditLog


class FedProxFairness(Strategy):
    """
    Custom FL strategy combining FedProx with fairness-weighted aggregation.
    
    FedProx: Adds proximal term to handle non-IID data
    Fairness Weighting: Balances AUROC, sample count, and domain relevance
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        proximal_mu: float = 0.01,
        fairness_weights: Dict[str, float] = None,
        domain_relevance_matrix: Dict[Tuple[str, str], float] = None,
        log_dir: str = "fl_results"
    ):
        """
        Initialize FedProxFairness strategy.
        
        Args:
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            proximal_mu: FedProx proximal term coefficient
            fairness_weights: Dict with keys 'auroc', 'samples', 'domain_relevance'
            domain_relevance_matrix: Pre-computed domain relevance scores
            log_dir: Directory for logging results
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.proximal_mu = proximal_mu
        
        # Fairness weights
        if fairness_weights is None:
            fairness_weights = {"auroc": 0.6, "samples": 0.3, "domain_relevance": 0.1}
        self.fairness_weights = fairness_weights
        
        # Domain relevance matrix
        self.domain_relevance_matrix = domain_relevance_matrix or {}
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Blockchain audit log
        self.blockchain_audit = BlockchainAuditLog(log_dir=str(self.log_dir / "blockchain_audit"))
        
        # Track metrics across rounds
        self.round_metrics = []
        
        print(f"+ FedProxFairness strategy initialized")
        print(f"  - Proximal µ: {self.proximal_mu}")
        print(f"  - Fairness weights: {self.fairness_weights}")
        print(f"  - Log directory: {self.log_dir}")
        print(f"  - Blockchain audit: Enabled")
    
    def compute_fairness_weight(
        self,
        client_id: str,
        auroc: float,
        num_samples: int,
        total_samples: int
    ) -> float:
        """
        Compute fairness weight for a client.
        
        Formula:
            weight = 0.6 * AUROC² + 0.3 * (samples / total) + 0.1 * avg_relevance
        
        Args:
            client_id: Hospital ID
            auroc: Client's AUROC score
            num_samples: Number of samples at this client
            total_samples: Total samples across all clients
        
        Returns:
            Fairness weight
        """
        # Component 1: AUROC squared
        auroc_component = self.fairness_weights["auroc"] * (auroc ** 2)
        
        # Component 2: Normalized sample count
        sample_component = self.fairness_weights["samples"] * (num_samples / total_samples)
        
        # Component 3: Average domain relevance to other clients
        relevance_scores = [
            score for (h1, h2), score in self.domain_relevance_matrix.items()
            if h1 == client_id and h2 != client_id
        ]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.5
        relevance_component = self.fairness_weights["domain_relevance"] * avg_relevance
        
        # Final weight
        weight = auroc_component + sample_component + relevance_component
        
        return weight
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients using fairness weighting.
        
        Args:
            server_round: Current FL round
            results: List of (client, fit_result) tuples
            failures: List of failed clients
        
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        print(f"\n{'='*60}")
        print(f"Round {server_round}: Aggregating {len(results)} clients")
        print(f"{'='*60}")
        
        # Extract metrics from results
        client_metrics = []
        for client, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics.get("hospital_id", "Unknown")
            auroc = metrics.get("auroc", 0.5)
            num_samples = fit_res.num_examples
            
            client_metrics.append({
                "client_id": client_id,
                "auroc": auroc,
                "num_samples": num_samples,
                "parameters": fit_res.parameters
            })
        
        # Compute total samples
        total_samples = sum(m["num_samples"] for m in client_metrics)
        
        # Compute fairness weights
        weights = []
        for m in client_metrics:
            weight = self.compute_fairness_weight(
                m["client_id"],
                m["auroc"],
                m["num_samples"],
                total_samples
            )
            weights.append(weight)
            
            print(f"  {m['client_id']}: AUROC={m['auroc']:.3f}, "
                  f"samples={m['num_samples']:,}, weight={weight:.4f}")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [float(w / total_weight) for w in weights]
        
        print(f"\nNormalized weights:")
        for m, norm_w in zip(client_metrics, normalized_weights):
            print(f"  {m['client_id']}: {norm_w:.4f} ({norm_w*100:.1f}%)")
        
        # Aggregate parameters using weighted average
        aggregated_params = self._weighted_average_parameters(
            [m["parameters"] for m in client_metrics],
            normalized_weights
        )
        
        # Log round metrics
        round_log = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "num_clients": len(results),
            "total_samples": total_samples,
            "clients": [
                {
                    "id": m["client_id"],
                    "auroc": m["auroc"],
                    "samples": m["num_samples"],
                    "raw_weight": float(w),
                    "normalized_weight": float(norm_w)
                }
                for m, w, norm_w in zip(client_metrics, weights, normalized_weights)
            ]
        }
        self.round_metrics.append(round_log)
        
        # Save round log
        log_path = self.log_dir / f"round_{server_round}_aggregation.json"
        with open(log_path, 'w') as f:
            json.dump(round_log, f, indent=2)
        
        # Log to blockchain audit
        self.blockchain_audit.log_fl_round(
            round_number=server_round,
            num_clients=len(results),
            total_samples=total_samples,
            client_weights=[
                {
                    "id": m["client_id"],
                    "auroc": m["auroc"],
                    "samples": m["num_samples"],
                    "normalized_weight": norm_w
                }
                for m, norm_w in zip(client_metrics, normalized_weights)
            ]
        )
        
        print(f"\n+ Aggregation complete. Log saved to: {log_path}")
        print(f"+ Blockchain audit updated (Block #{len(self.blockchain_audit.chain)-1})")
        
        # Return aggregated parameters and metrics
        metrics_aggregated = {
            "num_clients": len(results),
            "total_samples": total_samples
        }
        
        return aggregated_params, metrics_aggregated
    
    def _weighted_average_parameters(
        self,
        parameters_list: List[Parameters],
        weights: List[float]
    ) -> Parameters:
        """
        Compute weighted average of parameters.
        
        Args:
            parameters_list: List of Parameters objects
            weights: List of weights (must sum to 1.0)
        
        Returns:
            Weighted average Parameters
        """
        # Convert Parameters to numpy arrays
        arrays_list = [
            fl.common.parameters_to_ndarrays(params)
            for params in parameters_list
        ]
        
        # Compute weighted average for each tensor
        averaged_arrays = []
        num_tensors = len(arrays_list[0])
        
        for i in range(num_tensors):
            # Get i-th tensor from all clients
            tensors = [arrays[i] for arrays in arrays_list]
            
            # Weighted average
            weighted_tensor = sum(w * t for w, t in zip(weights, tensors))
            averaged_arrays.append(weighted_tensor)
        
        # Convert back to Parameters
        averaged_params = fl.common.ndarrays_to_parameters(averaged_arrays)
        
        return averaged_params
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics from clients.
        
        Args:
            server_round: Current FL round
            results: List of (client, evaluate_result) tuples
            failures: List of failed clients
        
        Returns:
            Aggregated loss and metrics
        """
        if not results:
            return None, {}
        
        # Weighted average of losses and AUROCs
        total_samples = sum(num_examples for _, eval_res in results for num_examples in [eval_res.num_examples])
        
        weighted_loss = sum(
            eval_res.num_examples * eval_res.loss
            for _, eval_res in results
        ) / total_samples
        
        # Extract AUROCs
        aurocs = []
        for _, eval_res in results:
            auroc = eval_res.metrics.get("auroc", 0.0)
            aurocs.append(auroc)
        
        avg_auroc = np.mean(aurocs)
        
        print(f"\nRound {server_round} Evaluation:")
        print(f"  Average Loss: {weighted_loss:.4f}")
        print(f"  Average AUROC: {avg_auroc:.4f}")
        
        metrics_aggregated = {
            "auroc": avg_auroc,
            "num_clients": len(results)
        }
        
        return weighted_loss, metrics_aggregated
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Create fit instructions with FedProx config
        fit_ins = fl.common.FitIns(
            parameters,
            {"proximal_mu": self.proximal_mu, "server_round": server_round}
        )
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Sample clients
        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Create evaluate instructions
        evaluate_ins = fl.common.EvaluateIns(parameters, {})
        
        return [(client, evaluate_ins) for client in clients]
    
    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        return None
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and min number of clients for training."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients
    
    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and min number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients


def main():
    """
    Start the enhanced FL server.
    """
    print("="*60)
    print("Enhanced Federated Learning Server")
    print("FedProx + Fairness Weighting + Differential Privacy")
    print("="*60)
    
    # Load configuration
    config = get_fl_config()
    print(f"\nConfiguration:")
    print(f"  Server address: {config['server_address']}")
    print(f"  Num rounds: {config['num_rounds']}")
    print(f"  FedProx mu: {config['fedprox_mu']}")
    print(f"  DP epsilon: {config['dp_epsilon']}, delta: {config['dp_delta']}")
    print(f"  Fairness weights: {config['fairness_weights']}")
    
    # Load domain relevance matrix
    print(f"\nLoading domain relevance matrix...")
    hospitals, _ = load_hospital_profiles(config["domain_relevance_config"])
    domain_relevance_matrix = compute_domain_relevance_matrix(
        hospitals,
        modality_weight=config["modality_weight"],
        label_weight=config["label_weight"],
        default_score=config["default_relevance"]
    )
    print(f"+ Loaded relevance scores for {len(hospitals)} hospitals")
    
    # Create strategy
    strategy = FedProxFairness(
        fraction_fit=config["fraction_fit"],
        fraction_evaluate=config["fraction_evaluate"],
        min_fit_clients=config["min_fit_clients"],
        min_evaluate_clients=config["min_evaluate_clients"],
        min_available_clients=config["min_available_clients"],
        proximal_mu=config["fedprox_mu"],
        fairness_weights=config["fairness_weights"],
        domain_relevance_matrix=domain_relevance_matrix,
        log_dir=config["log_dir"]
    )
    
    # Start server
    print(f"\n{'='*60}")
    print("Starting FL Server...")
    print(f"Waiting for clients to connect on {config['server_address']}")
    print(f"{'='*60}\n")
    
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        strategy=strategy
    )
    
    print(f"\n{'='*60}")
    print("FL Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {config['log_dir']}")


if __name__ == "__main__":
    main()
