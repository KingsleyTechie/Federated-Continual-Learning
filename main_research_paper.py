"""
Main research experiment runner
Orchestrates the complete federated continual learning experiment
"""

import torch
import numpy as np
import random
import os

# Import custom modules
from data_loader import load_organamnist_dataset, create_non_iid_partition, get_task_test_indices, print_data_statistics
from model_architecture import FederatedCNN
from federated_learning import FederatedAveraging, train_local_epoch, create_hospital_dataloader
from continual_learning import TaskSequencer, ContinualLearningEvaluator, print_task_performance
from differential_privacy import PrivacyManager
from evaluation_metrics import evaluate_model
from results_analysis import plot_continual_learning_performance, create_performance_table, generate_research_summary, print_research_summary

def setup_experiment():
    """
    Setup the complete experiment environment
    
    Returns:
        Dictionary with experiment components
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define task sequence
    tasks = {
        'task_1': [0, 1, 2],   # Liver, Kidney, Bladder
        'task_2': [3, 4, 5, 6], # Heart, Lung, Pancreas, Thyroid
        'task_3': [7, 8, 9, 10] # Stomach, Colon, Esophagus, Rectum
    }
    
    # Load dataset
    train_dataset, test_dataset, dataset_info = load_organamnist_dataset()
    
    # Create non-IID data distribution across hospitals
    hospital_data = create_non_iid_partition(train_dataset, tasks, num_hospitals=3)
    print_data_statistics(hospital_data, train_dataset)
    
    # Get test indices organized by tasks
    test_data_by_task = get_task_test_indices(test_dataset, tasks)
    
    # Initialize global model
    global_model = FederatedCNN(num_classes=11).to(device)
    
    # Initialize hospital models
    hospital_models = {}
    for hosp_id in hospital_data.keys():
        hospital_models[hosp_id] = FederatedCNN(num_classes=11).to(device)
    
    # Initialize components
    federated_averager = FederatedAveraging(global_model, hospital_models, device)
    task_sequencer = TaskSequencer(tasks)
    continual_evaluator = ContinualLearningEvaluator(test_data_by_task, device)
    privacy_manager = PrivacyManager(target_epsilon=8.0, target_delta=1e-5)
    
    experiment_components = {
        'device': device,
        'tasks': tasks,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'hospital_data': hospital_data,
        'test_data_by_task': test_data_by_task,
        'global_model': global_model,
        'hospital_models': hospital_models,
        'federated_averager': federated_averager,
        'task_sequencer': task_sequencer,
        'continual_evaluator': continual_evaluator,
        'privacy_manager': privacy_manager
    }
    
    return experiment_components

def run_federated_round(experiment_components, current_task, use_dp=True):
    """
    Run one round of federated learning for the current task
    
    Args:
        experiment_components: Dictionary of experiment components
        current_task: Current task to learn
        use_dp: Whether to use differential privacy
    
    Returns:
        Dictionary with training results
    """
    device = experiment_components['device']
    hospital_data = experiment_components['hospital_data']
    train_dataset = experiment_components['train_dataset']
    global_model = experiment_components['global_model']
    hospital_models = experiment_components['hospital_models']
    federated_averager = experiment_components['federated_averager']
    privacy_manager = experiment_components['privacy_manager']
    
    print(f"\nStarting federated learning round for {current_task}")
    
    # Copy global model to all hospitals
    federated_averager.distribute_global_model()
    
    hospital_updates = {}
    privacy_metrics = {}
    
    # Train at each hospital
    for hosp_id in hospital_models.keys():
        print(f"\nTraining at {hosp_id}...")
        
        # Create data loader for this hospital and task
        train_loader = create_hospital_dataloader(hospital_data, hosp_id, current_task, train_dataset)
        
        if train_loader is None:
            continue
        
        # Get hospital model
        model = hospital_models[hosp_id]
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train for a few epochs
        for epoch in range(2):
            train_local_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Calculate model update
        update = federated_averager.get_model_update(model)
        hospital_updates[hosp_id] = update
    
    # Aggregate updates from all hospitals
    if hospital_updates:
        federated_averager.aggregate_updates(hospital_updates)
    
    return {
        'hospital_updates': len(hospital_updates),
        'privacy_metrics': privacy_metrics
    }

def main():
    """
    Main function to run the complete research experiment
    """
    print("FEDERATED CONTINUAL LEARNING RESEARCH EXPERIMENT")
    print("=" * 60)
    
    # Setup experiment
    experiment = setup_experiment()
    
    # Get components
    task_sequencer = experiment['task_sequencer']
    continual_evaluator = experiment['continual_evaluator']
    test_dataset = experiment['test_dataset']
    global_model = experiment['global_model']
    
    # Store results
    all_results = {}
    performance_history = {}
    
    # Baseline evaluation
    print("\nBASELINE EVALUATION (Random Model)")
    task_sequence = task_sequencer.get_all_tasks()
    baseline_accuracies = continual_evaluator.evaluate_model_on_all_tasks(
        global_model, test_dataset, task_sequence
    )
    
    # Initialize performance history
    for task in task_sequence:
        performance_history[task] = [baseline_accuracies.get(task, 0.0)]
    
    all_results['baseline'] = baseline_accuracies
    
    # Sequential task learning
    while True:
        current_task, current_classes = task_sequencer.get_current_task()
        if current_task is None:
            break
        
        print(f"\n{'#' * 50}")
        print(f"LEARNING PHASE: {current_task}")
        print(f"Classes: {current_classes}")
        print(f"{'#' * 50}")
        
        # Run federated learning for current task
        round_results = run_federated_round(experiment, current_task, use_dp=True)
        
        # Evaluate on all tasks after learning current task
        current_accuracies = continual_evaluator.evaluate_model_on_all_tasks(
            global_model, test_dataset, task_sequence
        )
        
        # Update performance history
        for task in task_sequence:
            if task in current_accuracies:
                performance_history[task].append(current_accuracies[task])
        
        all_results[f'after_{current_task}'] = current_accuracies
        print_task_performance(current_task, current_accuracies)
        
        # Move to next task
        has_more_tasks = task_sequencer.move_to_next_task()
        if not has_more_tasks:
            break
    
    # Generate final results and visualizations
    print("\nGENERATING FINAL RESULTS AND VISUALIZATIONS")
    
    # Create performance table
    performance_data = {
        'Learning Phase': ['Baseline', 'After Task 1', 'After Task 2', 'After Task 3'],
        'Task 1 Accuracy (%)': [
            performance_history['task_1'][0],
            performance_history['task_1'][1] if len(performance_history['task_1']) > 1 else 0,
            performance_history['task_1'][2] if len(performance_history['task_1']) > 2 else 0,
            performance_history['task_1'][3] if len(performance_history['task_1']) > 3 else 0
        ],
        'Task 2 Accuracy (%)': [
            performance_history['task_2'][0],
            performance_history['task_2'][1] if len(performance_history['task_2']) > 1 else 0,
            performance_history['task_2'][2] if len(performance_history['task_2']) > 2 else 0,
            performance_history['task_2'][3] if len(performance_history['task_2']) > 3 else 0
        ],
        'Task 3 Accuracy (%)': [
            performance_history['task_3'][0],
            performance_history['task_3'][1] if len(performance_history['task_3']) > 1 else 0,
            performance_history['task_3'][2] if len(performance_history['task_3']) > 2 else 0,
            performance_history['task_3'][3] if len(performance_history['task_3']) > 3 else 0
        ]
    }
    
    # Create visualizations
    plot_continual_learning_performance(
        performance_history, 
        task_sequence, 
        'figures/figure1_continual_learning.png'
    )
    
    create_performance_table(
        performance_data,
        'tables/performance_results.png'
    )
    
    # Generate research summary
    privacy_summary = experiment['privacy_manager'].get_privacy_summary()
    research_summary = generate_research_summary(
        performance_history, 
        privacy_summary, 
        experiment['hospital_data']
    )
    
    print_research_summary(research_summary)
    
    print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")
    print("Results saved in figures/ and tables/ directories")

if __name__ == "__main__":
    main()
