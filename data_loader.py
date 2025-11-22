"""
Data loading and preprocessing module for medical imaging dataset
Handles OrganAMNIST dataset loading and partitioning for federated learning
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def load_organamnist_dataset():
    """
    Load the OrganAMNIST dataset with proper transformations
    
    Returns:
        train_dataset: Training dataset object
        test_dataset: Test dataset object
        info: Dataset information dictionary
    """
    # Dataset information
    data_flag = 'organamnist'
    info = INFO[data_flag]
    n_classes = len(info['label'])
    
    # Define image transformations
    # Normalize to [-1, 1] range for better training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load the dataset
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)
    
    print(f"Loaded OrganAMNIST dataset:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {n_classes}")
    
    return train_dataset, test_dataset, info

def create_non_iid_partition(dataset, tasks, num_hospitals=3):
    """
    Create non-IID data distribution across hospitals
    Each hospital specializes in different classes and has varying data quantities
    
    Args:
        dataset: PyTorch dataset object
        tasks: Dictionary mapping task names to class lists
        num_hospitals: Number of hospitals to partition data for
    
    Returns:
        hospital_data: Dictionary containing indices for each hospital and task
    """
    hospital_data = {}
    for i in range(num_hospitals):
        hospital_data[f'hospital_{i}'] = {task: [] for task in tasks.keys()}
    
    # Define hospital specializations - each hospital focuses on different classes
    hospital_specializations = {
        'hospital_0': {0: 0.6, 1: 0.2, 2: 0.2,  # Specializes in class 0 (liver)
                       3: 0.1, 4: 0.3, 5: 0.3, 6: 0.3,
                       7: 0.2, 8: 0.2, 9: 0.3, 10: 0.3},
        
        'hospital_1': {0: 0.2, 1: 0.6, 2: 0.2,  # Specializes in class 1 (kidney)
                       3: 0.3, 4: 0.1, 5: 0.3, 6: 0.3,
                       7: 0.3, 8: 0.2, 9: 0.2, 10: 0.3},
        
        'hospital_2': {0: 0.2, 1: 0.2, 2: 0.6,  # Specializes in class 2 (bladder)
                       3: 0.3, 4: 0.3, 5: 0.1, 6: 0.3,
                       7: 0.3, 8: 0.3, 9: 0.2, 10: 0.2}
    }
    
    # Collect samples for each class
    class_samples = {}
    for class_id in range(11):  # 11 classes in OrganAMNIST
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        class_samples[class_id] = class_indices
    
    # Distribute samples according to specializations
    for class_id in range(11):
        indices = class_samples[class_id]
        if len(indices) > 0:
            np.random.shuffle(indices)
            
            # Determine which task this class belongs to
            task_name = None
            for task, classes in tasks.items():
                if class_id in classes:
                    task_name = task
                    break
            
            if task_name:
                # Calculate total weight for normalization
                total_weight = sum([hospital_specializations[hosp][class_id] 
                                  for hosp in hospital_specializations])
                
                start_idx = 0
                for hosp_id, hosp_data in hospital_data.items():
                    hosp_weight = hospital_specializations[hosp_id][class_id]
                    num_samples = int(len(indices) * (hosp_weight / total_weight))
                    
                    if num_samples > 0:
                        end_idx = min(start_idx + num_samples, len(indices))
                        selected_indices = indices[start_idx:end_idx]
                        hosp_data[task_name].extend(selected_indices)
                        start_idx = end_idx
    
    return hospital_data

def get_task_test_indices(test_dataset, tasks):
    """
    Get test indices organized by tasks for evaluation
    
    Args:
        test_dataset: Test dataset object
        tasks: Dictionary mapping task names to class lists
    
    Returns:
        test_data_by_task: Dictionary containing test indices for each task
    """
    test_data_by_task = {}
    for task_name, classes in tasks.items():
        task_indices = [i for i in range(len(test_dataset)) 
                       if test_dataset[i][1] in classes]
        test_data_by_task[task_name] = task_indices
    
    return test_data_by_task

def print_data_statistics(hospital_data, train_dataset):
    """
    Print statistics about the data distribution across hospitals
    
    Args:
        hospital_data: Dictionary containing hospital data partitions
        train_dataset: Training dataset for label information
    """
    print("\nHospital Data Statistics:")
    for hosp_id, tasks_data in hospital_data.items():
        total_samples = 0
        print(f"\n{hosp_id}:")
        for task_name, indices in tasks_data.items():
            if indices:
                labels = [train_dataset[i][1] for i in indices]
                unique, counts = np.unique(labels, return_counts=True)
                task_samples = len(indices)
                total_samples += task_samples
                class_dist = {cls: count for cls, count in zip(unique, counts)}
                print(f"  {task_name}: {task_samples} samples")
                print(f"    Class distribution: {class_dist}")
        print(f"  TOTAL: {total_samples} samples")
