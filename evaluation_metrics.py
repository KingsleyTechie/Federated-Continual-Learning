"""
Evaluation metrics and model testing functions
Provides comprehensive evaluation of model performance
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device, task_name="All Tasks"):
    """
    Comprehensive model evaluation on a test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        task_name: Name of the task for logging
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device).squeeze()
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    
    print(f'Test Accuracy on {task_name}: {accuracy:.2f}%')
    
    return accuracy

def calculate_confusion_matrix(model, test_loader, device, class_names=None):
    """
    Calculate and return confusion matrix
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: List of class names for labeling
    
    Returns:
        conf_matrix: Confusion matrix as numpy array
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device).squeeze()
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def calculate_class_wise_accuracy(model, test_loader, device, num_classes):
    """
    Calculate accuracy for each class separately
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        num_classes: Number of classes
    
    Returns:
        class_accuracies: Dictionary of accuracies per class
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device).squeeze()
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            # Update class-wise statistics
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # Calculate accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            class_accuracies[i] = accuracy
    
    return class_accuracies

def print_detailed_evaluation(model, test_loader, device, class_names):
    """
    Print detailed evaluation results including class-wise performance
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: List of class names
    """
    # Overall accuracy
    overall_accuracy = evaluate_model(model, test_loader, device)
    
    # Class-wise accuracy
    class_accuracies = calculate_class_wise_accuracy(model, test_loader, device, len(class_names))
    
    print("\nClass-wise Performance:")
    for class_id, accuracy in class_accuracies.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"  {class_name}: {accuracy:.2f}%")
    
    return overall_accuracy, class_accuracies
