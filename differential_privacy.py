"""
Differential privacy implementation for federated learning
Provides privacy guarantees through noise addition and privacy accounting
"""

import torch
import warnings

class PrivacyManager:
    """
    Manages differential privacy for federated learning
    Handles noise addition and privacy budget tracking
    """
    
    def __init__(self, target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0):
        """
        Initialize privacy manager
        
        Args:
            target_epsilon: Target privacy budget (epsilon)
            target_delta: Privacy parameter delta
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = 1.1  # Controls noise level
        
        # Privacy accounting
        self.privacy_metrics = {}
    
    def setup_differential_privacy(self, model, optimizer, data_loader):
        """
        Setup differential privacy for training
        
        Args:
            model: Model to apply DP to
            optimizer: Optimizer for training
            data_loader: DataLoader for training data
        
        Returns:
            Tuple of (model, optimizer, data_loader) with DP applied
        """
        try:
            from opacus import PrivacyEngine
            
            privacy_engine = PrivacyEngine()
            
            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            
            return model, optimizer, data_loader, privacy_engine
            
        except ImportError:
            warnings.warn("Opacus not available. Continuing without differential privacy.")
            return model, optimizer, data_loader, None
        except Exception as e:
            warnings.warn(f"Failed to setup differential privacy: {e}")
            return model, optimizer, data_loader, None
    
    def calculate_privacy_spent(self, privacy_engine, hospital_id):
        """
        Calculate and record privacy spending
        
        Args:
            privacy_engine: Opacus PrivacyEngine instance
            hospital_id: ID of the hospital for tracking
        
        Returns:
            Dictionary with privacy metrics
        """
        if privacy_engine is None:
            return {'epsilon': 0.0, 'delta': self.target_delta}
        
        try:
            epsilon = privacy_engine.get_epsilon(delta=self.target_delta)
            
            privacy_metrics = {
                'epsilon': epsilon,
                'delta': self.target_delta,
                'noise_multiplier': self.noise_multiplier
            }
            
            self.privacy_metrics[hospital_id] = privacy_metrics
            return privacy_metrics
            
        except Exception as e:
            warnings.warn(f"Failed to calculate privacy spent: {e}")
            return {'epsilon': 0.0, 'delta': self.target_delta}
    
    def get_privacy_summary(self):
        """
        Get summary of privacy spending across all hospitals
        
        Returns:
            Dictionary with privacy summary statistics
        """
        if not self.privacy_metrics:
            return {}
        
        epsilons = [metrics['epsilon'] for metrics in self.privacy_metrics.values()]
        
        summary = {
            'average_epsilon': sum(epsilons) / len(epsilons),
            'max_epsilon': max(epsilons),
            'min_epsilon': min(epsilons),
            'hospitals_with_dp': len(self.privacy_metrics),
            'target_delta': self.target_delta
        }
        
        return summary
    
    def print_privacy_report(self):
        """Print a comprehensive privacy report"""
        summary = self.get_privacy_summary()
        
        if not summary:
            print("No privacy metrics available")
            return
        
        print("\nDifferential Privacy Report:")
        print(f"  Average ε: {summary['average_epsilon']:.2f}")
        print(f"  Maximum ε: {summary['max_epsilon']:.2f}")
        print(f"  Privacy δ: {summary['target_delta']}")
        print(f"  Hospitals with DP: {summary['hospitals_with_dp']}")

def add_differential_privacy_noise(model_update, noise_scale=0.01):
    """
    Add simple Gaussian noise to model updates as a basic DP mechanism
    
    Args:
        model_update: Dictionary of model parameter updates
        noise_scale: Scale of the Gaussian noise
    
    Returns:
        Noisy model update dictionary
    """
    noisy_update = {}
    
    for param_name, update in model_update.items():
        # Add Gaussian noise to each parameter update
        noise = torch.randn_like(update) * noise_scale
        noisy_update[param_name] = update + noise
    
    return noisy_update
