"""
Continual learning components for sequential task training
Handles task sequencing and manages learning across multiple tasks
"""

import torch
import numpy as np

class TaskSequencer:
    """
    Manages the sequence of tasks for continual learning
    Handles task transitions and task-specific data management
    """
    
    def __init__(self, tasks_definition):
        """
        Initialize task sequencer with task definitions
        
        Args:
            tasks_definition: Dictionary mapping task names to class lists
        """
        self.tasks_definition = tasks_definition
        self.task_sequence = list(tasks_definition.keys())
        self.current_task_index = 0
        
    def get_current_task(self):
        """
        Get the current active task
        
        Returns:
            Current task name and classes
        """
        if self.current_task_index < len(self.task_sequence):
            task_name = self.task_sequence[self.current_task_index]
            return task_name, self.tasks_definition[task_name]
        return None, None
    
    def move_to_next_task(self):
        """
        Advance to the next task in the sequence
        
        Returns:
            Boolean indicating if more tasks remain
        """
        self.current_task_index += 1
        return self.current_task_index < len(self.task_sequence)
    
    def get_all_tasks(self):
        """
        Get all tasks in the sequence
        
        Returns:
            List of all task names
        """
        return self.task_sequence
    
    def reset_sequence(self):
        """Reset task sequence to beginning"""
        self.current_task_index = 0

class ContinualLearningEvaluator:
    """
    Evaluates model performance across all tasks to measure forgetting
    """
    
    def __init__(self, test_data_by_task, device):
        """
        Initialize continual learning evaluator
        
        Args:
            test_data_by_task: Dictionary of test indices for each task
            device: Device to run evaluation on
        """
        self.test_data_by_task = test_data_by_task
        self.device = device
        self.performance_history = {}
    
    def evaluate_model_on_all_tasks(self, model, test_dataset, task_sequence):
        """
        Evaluate model performance on all tasks
        
        Args:
            model: Model to evaluate
            test_dataset: Full test dataset
            task_sequence: List of tasks to evaluate on
        
        Returns:
            Dictionary of accuracies for each task
        """
        model.eval()
        task_accuracies = {}
        
        for task_name in task_sequence:
            if task_name not in self.test_data_by_task:
                continue
                
            task_indices = self.test_data_by_task[task_name]
            if not task_indices:
                task_accuracies[task_name] = 0.0
                continue
            
            # Create DataLoader for this task's test data
            from torch.utils.data import DataLoader, Subset
            task_test_dataset = Subset(test_dataset, task_indices)
            test_loader = DataLoader(task_test_dataset, batch_size=64, shuffle=False)
            
            # Calculate accuracy for this task
            accuracy = self._calculate_accuracy(model, test_loader)
            task_accuracies[task_name] = accuracy
            
            # Update performance history
            if task_name not in self.performance_history:
                self.performance_history[task_name] = []
            self.performance_history[task_name].append(accuracy)
        
        return task_accuracies
    
    def _calculate_accuracy(self, model, data_loader):
        """
        Calculate accuracy for a given data loader
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
        
        Returns:
            Accuracy percentage
        """
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device).squeeze()
                
                outputs = model(data)
                _, predicted = outputs.max(1)
                total_samples += targets.size(0)
                correct_predictions += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct_predictions / total_samples
        return accuracy
    
    def calculate_forgetting_measure(self):
        """
        Calculate forgetting measure for each task
        
        Returns:
            Dictionary of forgetting measures for each task
        """
        forgetting_measures = {}
        
        for task_name, history in self.performance_history.items():
            if len(history) >= 2:
                # Forgetting is the difference between peak performance and final performance
                peak_performance = max(history)
                final_performance = history[-1]
                forgetting = peak_performance - final_performance
                forgetting_measures[task_name] = forgetting
        
        return forgetting_measures

def print_task_performance(current_task, task_accuracies):
    """
    Print performance results for the current learning phase
    
    Args:
        current_task: Name of the current task being learned
        task_accuracies: Dictionary of accuracies for all tasks
    """
    print(f"\nPerformance after learning {current_task}:")
    for task_name, accuracy in task_accuracies.items():
        print(f"  {task_name}: {accuracy:.2f}%")
