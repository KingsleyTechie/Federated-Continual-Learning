"""
Results analysis and visualization functions
Generates plots and tables for research paper
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import os

def plot_continual_learning_performance(performance_history, task_sequence, save_path=None):
    """
    Plot continual learning performance across tasks
    
    Args:
        performance_history: Dictionary of performance lists for each task
        task_sequence: List of task names in order
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare data for plotting
    learning_stages = ['Baseline', 'After Task 1', 'After Task 2', 'After Task 3']
    
    # Plot 1: Accuracy progression
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    task_labels = ['Task 1', 'Task 2', 'Task 3']
    
    x = np.arange(len(learning_stages))
    width = 0.25
    
    for i, task in enumerate(task_sequence):
        if task in performance_history:
            accuracies = performance_history[task]
            # Pad with zeros if needed
            while len(accuracies) < len(learning_stages):
                accuracies.append(0)
            ax1.bar(x + (i-1)*width, accuracies, width, label=task_labels[i], 
                   color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Learning Stage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Continual Learning Performance Across Tasks', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(learning_stages, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forgetting analysis
    if len(performance_history) >= 2:
        task1_forgetting = [0, 0]
        if len(performance_history[task_sequence[0]]) >= 3:
            peak_task1 = performance_history[task_sequence[0]][1]  # After Task 1
            final_task1 = performance_history[task_sequence[0]][-1]  # Final
            task1_forgetting.extend([peak_task1 - final_task1] * 2)
        
        stages_for_plot = learning_stages[2:]  # Start from After Task 2
        
        ax2.plot(stages_for_plot, task1_forgetting[2:], 'o-', linewidth=3, 
                markersize=8, label='Task 1 Forgetting', color=colors[0])
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Learning Stage', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy Loss (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Catastrophic Forgetting Analysis', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved continual learning plot to: {save_path}")
    
    return fig

def create_performance_table(performance_data, save_path=None):
    """
    Create a performance table for publication
    
    Args:
        performance_data: Dictionary of performance metrics
        save_path: Path to save the table (optional)
    """
    # Create DataFrame
    df = pd.DataFrame(performance_data)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code cells
    for i in range(len(df.columns)):
        for j in range(len(df)):
            if i > 0:  # Skip first column
                value = df.iloc[j, i]
                if isinstance(value, (int, float)):
                    if value == 0:
                        table[(j+1, i)].set_facecolor('#FFCCCC')
                    elif value > 20:
                        table[(j+1, i)].set_facecolor('#CCFFCC')
                    elif value > 0:
                        table[(j+1, i)].set_facecolor('#FFFFCC')
    
    # Header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Federated Continual Learning Performance', 
             pad=20, fontsize=14, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved performance table to: {save_path}")
    
    return fig

def generate_research_summary(performance_history, privacy_summary, hospital_data):
    """
    Generate a comprehensive research summary
    
    Args:
        performance_history: Performance data across tasks
        privacy_summary: Privacy metrics summary
        hospital_data: Hospital data statistics
    
    Returns:
        Dictionary with research summary
    """
    summary = {
        'performance': {},
        'privacy': privacy_summary,
        'data_statistics': {}
    }
    
    # Performance summary
    for task, history in performance_history.items():
        if history:
            summary['performance'][task] = {
                'initial': history[0] if len(history) > 0 else 0,
                'peak': max(history) if history else 0,
                'final': history[-1] if history else 0
            }
    
    # Data statistics
    total_samples = 0
    for hosp_id, tasks_data in hospital_data.items():
        hosp_samples = sum(len(indices) for indices in tasks_data.values())
        total_samples += hosp_samples
        summary['data_statistics'][hosp_id] = hosp_samples
    
    summary['data_statistics']['total'] = total_samples
    
    return summary

def print_research_summary(summary):
    """
    Print a formatted research summary
    
    Args:
        summary: Research summary dictionary
    """
    print("\n" + "="*60)
    print("RESEARCH SUMMARY")
    print("="*60)
    
    print("\nPERFORMANCE RESULTS:")
    for task, metrics in summary['performance'].items():
        print(f"  {task}:")
        print(f"    Initial: {metrics['initial']:.2f}%")
        print(f"    Peak: {metrics['peak']:.2f}%")
        print(f"    Final: {metrics['final']:.2f}%")
        if metrics['peak'] > 0:
            forgetting = metrics['peak'] - metrics['final']
            print(f"    Forgetting: {forgetting:+.2f}%")
    
    print("\nPRIVACY METRICS:")
    if summary['privacy']:
        for metric, value in summary['privacy'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\nDATA STATISTICS:")
    for hosp, samples in summary['data_statistics'].items():
        print(f"  {hosp}: {samples} samples")
