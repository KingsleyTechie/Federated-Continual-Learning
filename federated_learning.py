"""
Federated learning components and aggregation logic
Implements federated averaging and client update management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy

class FederatedAveraging:
    """
    Implements the Federated Averaging algorithm for model aggregation
    """
    
    def __init__(self, global_model, hospital_models, device):
        """
        Initialize the federated averaging coordinator
        
        Args:
            global_model: The global model that gets updated
            hospital_models: Dictionary of hospital-specific models
            device: Device to run computations on
        """
        self.global_model = global_model
        self.hospital_models = hospital_models
        self.device = device
    
    def aggregate_updates(self, hospital_updates):
        """
        Aggregate hospital updates using federated averaging
        
        Args:
            hospital_updates: Dictionary of model updates from hospitals
        
        Returns:
            None (updates global model in-place)
        """
        if not hospital_updates:
            print("No hospital updates to aggregate")
            return
        
        with torch.no_grad():
            # Initialize averaged update dictionary
            averaged_update = {}
            
            # Get parameter names from first hospital
            first_hosp = list(hospital_updates.keys())[0]
            param_names = hospital_updates[first_hosp].keys()
            
            # Average updates for each parameter
            for param_name in param_names:
                updates = []
                for hosp_id in hospital_updates:
                    if param_name in hospital_updates[hosp_id]:
                        updates.append(hospital_updates[hosp_id][param_name])
                
                if updates:
                    # Stack and average the updates
                    stacked_updates = torch.stack(updates)
                    averaged_update[param_name] = torch.mean(stacked_updates, dim=0)
            
            # Apply averaged update to global model
            for name, param in self.global_model.named_parameters():
                if name in averaged_update:
                    param.data += averaged_update[name]
        
        print(f"Aggregated updates from {len(hospital_updates)} hospitals")
    
    def get_model_update(self, local_model):
        """
        Calculate the difference between local and global model
        
        Args:
            local_model: Hospital's locally trained model
        
        Returns:
            update_dict: Dictionary of parameter updates
        """
        update_dict = {}
        
        # Iterate through both models simultaneously
        for (global_name, global_param), (local_name, local_param) in \
            zip(self.global_model.named_parameters(), local_model.named_parameters()):
            
            # Ensure we're comparing the same parameters
            if global_name == local_name:
                update_dict[global_name] = local_param.data - global_param.data
        
        return update_dict
    
    def distribute_global_model(self):
        """
        Copy global model weights to all hospital models
        """
        global_state_dict = self.global_model.state_dict()
        
        for hosp_id, model in self.hospital_models.items():
            model.load_state_dict(global_state_dict)
        
        print("Distributed global model to all hospitals")

def train_local_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train a model for one epoch locally at a hospital
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on
        epoch: Current epoch number for logging
    
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        targets = targets.to(device).squeeze()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        correct_predictions += predicted.eq(targets).sum().item()
    
    # Calculate epoch metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    # Print progress every few epochs
    if epoch % 2 == 0:
        print(f'Epoch {epoch}: Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

def create_hospital_dataloader(hospital_data, hospital_id, task_name, train_dataset, batch_size=32):
    """
    Create DataLoader for a specific hospital and task
    
    Args:
        hospital_data: Dictionary containing all hospital data
        hospital_id: ID of the hospital
        task_name: Name of the task to train on
        train_dataset: Full training dataset
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader for the specified hospital and task, or None if no data
    """
    task_indices = hospital_data[hospital_id][task_name]
    
    if not task_indices:
        print(f"No data for {task_name} at {hospital_id}")
        return None
    
    hospital_dataset = Subset(train_dataset, task_indices)
    dataloader = DataLoader(hospital_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
