"""
Configuration file for the Tox21 modeling pipeline.

This file centralizes all configuration settings for easy maintenance and modification.
"""

import logging

# Pipeline Configuration
PIPELINE_CONFIG = {
    # General settings
    'random_state': 42,
    'cv_folds': 5,
    'resume': True,
    'log_level': logging.INFO,

    # Data requirements
    'min_samples': 100,
    'min_positive_samples': 10,

    # Feature selection settings
    'feature_selection': {
        'correlation_threshold': 0.90,
        'univariate_k': 500,
        'top_n_model': 150
    },

    # Models to train
    'models': ['RandomForest', 'LogisticRegression', 'SVM'],

    # Model-specific configurations
    'model_configs': {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced'
        },
        'LogisticRegression': {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced'
        }
    },

    # File paths
    'paths': {
        'results_dir': 'results',
        'logs_dir': 'logs',
        'data_dir': 'data',
        'descriptors_file': 'results/tox21_descriptors.npy'
    }
}

# Target-specific configurations (if needed)
TARGET_CONFIGS = {
    'NR-AR': {
        'feature_selection': {
            'correlation_threshold': 0.85,
            'univariate_k': 400,
            'top_n_model': 120
        }
    },
    'SR-ARE': {
        'feature_selection': {
            'correlation_threshold': 0.95,
            'univariate_k': 600,
            'top_n_model': 180
        }
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_handler': True,
    'console_handler': True
}

# Evaluation metrics
EVALUATION_METRICS = [
    'roc_auc',
    'pr_auc',
    'f1',
    'precision',
    'recall',
    'accuracy',
    'balanced_accuracy'
]

# Default target indices for quick testing
DEFAULT_TARGET_INDICES = [0, 7]  # NR-AR, SR-ARE

# All available targets (for reference)
ALL_TARGETS = [
    'NR-AR',      # Nuclear Receptor Androgen Receptor
    'NR-AR-LBD',  # Nuclear Receptor Androgen Receptor Ligand Binding Domain
    'NR-AhR',     # Nuclear Receptor Aryl Hydrocarbon Receptor
    'NR-Aromatase', # Nuclear Receptor Aromatase
    'NR-ER',      # Nuclear Receptor Estrogen Receptor
    'NR-ER-LBD',  # Nuclear Receptor Estrogen Receptor Ligand Binding Domain
    'NR-PPAR-gamma', # Nuclear Receptor Peroxisome Proliferator Activated Receptor Gamma
    'SR-ARE',     # Stress Response Antioxidant Response Element
    'SR-ATAD5',   # Stress Response DNA Damage Response
    'SR-HSE',     # Stress Response Heat Shock Response
    'SR-MMP',     # Stress Response Matrix Metalloproteinase
    'SR-p53'      # Stress Response p53 Response
]