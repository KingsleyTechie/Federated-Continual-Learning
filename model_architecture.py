"""
Neural network architecture definitions
Contains the FederatedCNN model for medical image classification
"""

import torch
import torch.nn as nn

class FederatedCNN(nn.Module):
    """
    Convolutional Neural Network for federated learning on medical images
    Designed for 28x28 grayscale images from OrganAMNIST dataset
    """
    
    def __init__(self, num_classes=11):
        """
        Initialize the FederatedCNN model
        
        Args:
            num_classes: Number of output classes (11 for OrganAMNIST)
        """
        super(FederatedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Regularization
            nn.Dropout(0.25)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 28x28 -> 7x7 after two max pools
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_with_sample_data(model, device):
    """
    Test the model with sample data to verify forward pass
    
    Args:
        model: Initialized model
        device: Device to run test on (cpu or cuda)
    """
    # Create sample batch
    batch_size = 4
    sample_input = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Model test completed:")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    return output.shape
