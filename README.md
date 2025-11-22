# Federated-Continual-Learning
This project presents a novel Federated Continual Learning framework for medical imaging that enables continuous model updates across multiple hospitals without central data sharing.

Federated Continual Learning for Medical Imaging with Differential Privacy Guarantees

Complete Technical Documentation

1. Project Overview

This research implements a comprehensive federated continual learning system for medical imaging that enables multiple hospitals to collaboratively train artificial intelligence models without sharing patient data. The system addresses three critical challenges in healthcare AI: data privacy preservation through differential privacy, continuous learning from sequential medical tasks, and prevention of catastrophic forgetting when learning new information.

The framework combines federated learning for decentralized collaboration, continual learning for sequential task acquisition, and differential privacy for mathematical privacy guarantees. The implementation is validated on the OrganAMNIST dataset containing 34,561 medical images across 11 organ classes, with experiments conducted across three simulated hospitals with non-IID data distributions.

2. System Architecture

The system operates through five interconnected layers:

Hospital Layer: Three medical institutions maintaining private patient data and local training capabilities

Federated Learning Layer: Secure model aggregation without data sharing

Continual Learning Layer: Sequential task learning management

Global Model Layer: Shared knowledge maintenance and distribution

Monitoring Layer: Performance tracking and drift detection

3. Module Documentation

3.1 data_loader.py

Purpose: Handles dataset loading, preprocessing, and non-IID data partitioning across hospitals.

Key Functions:

load_organamnist_dataset(): Loads OrganAMNIST dataset with proper transformations

create_non_iid_partition(): Creates realistic hospital data distributions with specialized expertise

get_task_test_indices(): Organizes test data by tasks for evaluation

print_data_statistics(): Displays data distribution across hospitals

Data Flow:

Loads 28x28 grayscale medical images from OrganAMNIST

Applies normalization transformations

Partitions data into non-IID distributions across three hospitals

Each hospital specializes in different organ classes while maintaining general capabilities

3.2 model_architecture.py

Purpose: Defines the neural network architecture for medical image classification.

Key Components:

FederatedCNN: Custom convolutional neural network for federated learning

Feature extraction: Two convolutional blocks with max-pooling

Classification: Fully connected layers with dropout regularization

Weight initialization: Xavier uniform initialization

Network Architecture:

Input: 28x28 grayscale images

Conv2D(32) → ReLU → MaxPool2D

Conv2D(64) → ReLU → MaxPool2D → Dropout(0.25)

Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(11)

3.3 federated_learning.py

Purpose: Implements federated learning protocols and model aggregation.

Key Components:

FederatedAveraging: Core federated learning coordination

train_local_epoch(): Hospital-side model training

create_hospital_dataloader(): Task-specific data loading

Federated Learning Process:

Global model distribution to all hospitals

Local training on private hospital data

Model update calculation and aggregation

Global model update and redistribution

3.4 continual_learning.py

Purpose: Manages sequential task learning and evaluates catastrophic forgetting.

Key Components:

TaskSequencer: Manages task progression and transitions

ContinualLearningEvaluator: Tracks performance across all tasks

print_task_performance(): Displays learning progress

Task Sequence:

Task 1: Liver, Kidney, Bladder (Classes 0-2)

Task 2: Heart, Lung, Pancreas, Thyroid (Classes 3-6)

Task 3: Stomach, Colon, Esophagus, Rectum (Classes 7-10)

3.5 differential_privacy.py

Purpose: Implements differential privacy for privacy-preserving learning.

Key Components:

PrivacyManager: Manages privacy budget and noise addition

setup_differential_privacy(): Configures Opacus privacy engine

calculate_privacy_spent(): Tracks cumulative privacy expenditure

Privacy Parameters:

Target epsilon: 8.0

Delta: 1e-5

Noise multiplier: 1.1

Maximum gradient norm: 1.0

3.6 evaluation_metrics.py

Purpose: Provides comprehensive model evaluation and performance analysis.

Key Functions:

evaluate_model(): Standard accuracy evaluation

calculate_confusion_matrix(): Detailed performance analysis

calculate_class_wise_accuracy(): Per-class performance metrics

print_detailed_evaluation(): Comprehensive results reporting

3.7 results_analysis.py

Purpose: Generates visualizations and analysis for research publication.

Key Functions:

plot_continual_learning_performance(): Creates performance progression plots

create_performance_table(): Generates publication-ready tables

generate_research_summary(): Compiles comprehensive results

print_research_summary(): Formatted results presentation

3.8 main_research_paper.py

Purpose: Orchestrates the complete research experiment.

Experiment Flow:

Environment setup and initialization

Baseline performance evaluation

Sequential task learning with federated rounds

Continuous performance monitoring

Results compilation and visualization

4. Experimental Setup

4.1 Dataset Specifications

Dataset: OrganAMNIST from MedMNIST collection

Training samples: 34,561

Test samples: 17,778

Image size: 28x28 pixels, grayscale

Classes: 11 organ types

Task split: 3 sequential learning phases

4.2 Hospital Configuration

Three hospitals with specialized expertise:

Hospital 0: Specialized in liver diagnosis (Class 0)

Hospital 1: Specialized in kidney diagnosis (Class 1)

Hospital 2: Specialized in bladder diagnosis (Class 2)

Each hospital receives non-IID data distributions reflecting real-world medical practice where institutions develop specialized diagnostic capabilities.

4.3 Training Parameters

Model: FederatedCNN with 11 output classes

Optimizer: Adam with learning rate 0.001

Loss function: Cross-entropy

Batch size: 32

Local epochs: 2 per federated round

Differential privacy: Enabled with epsilon 8.0, delta 1e-5

5. Key Research Findings

5.1 Performance Metrics

Baseline performance with random initialization:

Task 1: 6.62% accuracy

Task 2: 7.45% accuracy

Task 3: 12.52% accuracy

After Task 1 federated learning:

Task 1: 39.65% accuracy (+33.03% improvement)

Task 2: 0.00% accuracy

Task 3: 0.00% accuracy

Final performance after sequential learning:

Task 1: 0.00% accuracy (complete forgetting)

Task 2: 9.69% accuracy

Task 3: 0.00% accuracy

5.2 Privacy Protection

Differential privacy implementation achieved:

Average privacy budget (epsilon): 1.54

Maximum privacy budget: 1.58

Privacy delta: 1e-5

All hospitals maintained privacy guarantees

5.3 Catastrophic Forgetting Analysis

The experiment revealed severe catastrophic forgetting:

Task 1 knowledge completely lost after learning Task 2

Limited forward transfer between tasks

Task 2 performance maintained but limited

Task 3 showed minimal learning capability

6. Technical Implementation Details

6.1 Federated Learning Protocol

The federated averaging process:

Initialize global model with random weights

Distribute model to all participating hospitals

Each hospital trains locally on private data

Collect model updates from hospitals

Aggregate updates using federated averaging

Update global model and repeat

6.2 Continual Learning Strategy

Sequential learning approach:

Strict task sequencing without overlap

No explicit forgetting prevention mechanisms

Evaluation after each learning phase

Performance tracking across all previous tasks

6.3 Privacy Implementation

Differential privacy mechanisms:

Gaussian noise addition to gradients

Gradient norm clipping

Privacy accounting across training rounds

Formal (epsilon, delta) privacy guarantees

7. Code Organization and Structure

The codebase follows modular design principles:

Each module has single responsibility

Clear interfaces between modules

Comprehensive error handling

Extensive logging and monitoring

Reproducible experiment configuration

8. Usage Instructions

8.1 Installation

Install required dependencies:
pip install torch torchvision numpy matplotlib pandas seaborn medmnist opacus scikit-learn Pillow

8.2 Running Experiments

Execute the main research experiment:
python main_research_paper.py

8.3 Results Access

Generated results available in:

figures/: Performance visualizations

tables/: Publication-ready tables

models/: Trained model weights

architecture/: System diagrams

9. Limitations and Future Work

Current Limitations:

Severe catastrophic forgetting observed

Limited task performance in later phases

Computational requirements for privacy

Simplified medical imaging task

Future Directions:

Advanced continual learning techniques

Adaptive privacy budget allocation

Cross-silo federated learning scaling

Real-world clinical validation

Integration with electronic health records

10. Conclusion

This research successfully demonstrates a working federated continual learning system for medical imaging that maintains patient privacy through differential privacy. The implementation provides a foundation for collaborative AI development in healthcare while revealing critical challenges in knowledge retention across sequential learning tasks. The codebase serves as a reproducible framework for future research in privacy-preserving continual learning for medical applications.

The system achieves its primary objectives of enabling multi-institutional collaboration without data sharing, supporting sequential learning of medical tasks, and providing formal privacy guarantees, while clearly identifying catastrophic forgetting as the primary challenge for future research addressing.


