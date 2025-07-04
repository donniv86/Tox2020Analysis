"""
Model Trainer - Handles model training and evaluation
"""

import time
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training and evaluation with cross-validation.

    Responsibilities:
    - Model initialization and configuration
    - Cross-validation training
    - Performance evaluation
    - Result aggregation
    """

    def __init__(self, models: List[str], cv_folds: int = 5, random_state: int = 42):
        """
        Initialize the model trainer.

        Args:
            models: List of model names to train
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.models = models
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model_configs = self._get_model_configs()

    def _get_model_configs(self) -> Dict[str, Any]:
        """Get model configurations."""
        return {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
            },
            'SVM': {
                'class': SVC,
                'params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
            }
        }

    def _create_model(self, model_name: str):
        """Create a model instance."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.model_configs[model_name]
        return config['class'](**config['params'])

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': (y_true == y_pred).mean(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba)
        }

        # Calculate balanced accuracy
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['balanced_accuracy'] = (tp / (tp + fn) + tn / (tn + fp)) / 2

        return metrics

    def _train_single_model(self, model_name: str, X: np.ndarray, y: np.ndarray, target_name: str) -> Dict[str, Any]:
        """Train a single model with cross-validation."""
        logger.info(f"Training {model_name} for {target_name}")
        start_time = time.time()

        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Initialize results storage
        fold_results = {
            'roc_auc': [], 'pr_auc': [], 'f1': [],
            'precision': [], 'recall': [], 'accuracy': [], 'balanced_accuracy': []
        }

        # Cross-validation loop
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            fold_start_time = time.time()

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and train model
            model = self._create_model(model_name)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Store results
            for metric, value in metrics.items():
                if metric in fold_results:
                    fold_results[metric].append(value)

            fold_time = time.time() - fold_start_time
            logger.debug(f"{model_name} - Fold {fold_idx + 1}/{self.cv_folds} completed in {fold_time:.1f}s")

        # Calculate average metrics
        avg_metrics = {}
        for metric, values in fold_results.items():
            avg_metrics[f'{metric}_mean'] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)

        training_time = time.time() - start_time

        # Create final model on full dataset
        final_model = self._create_model(model_name)
        final_model.fit(X, y)

        return {
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'model': final_model,
            'training_time': training_time
        }

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, target_name: str) -> Dict[str, Any]:
        """
        Train and evaluate all models for a target.

        Args:
            X: Feature matrix
            y: Target vector
            target_name: Name of the target

        Returns:
            Dictionary containing results for all models
        """
        logger.info(f"Training {len(self.models)} models for {target_name}")

        results = {}
        model_training_times = {}

        # Train each model
        for model_name in self.models:
            try:
                model_results = self._train_single_model(model_name, X, y, target_name)
                results[model_name] = model_results
                model_training_times[model_name] = model_results['training_time']

                # Log performance
                avg_metrics = model_results['avg_metrics']
                logger.info(f"{model_name} - ROC-AUC: {avg_metrics['roc_auc_mean']:.3f}±{avg_metrics['roc_auc_std']:.3f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        # Find best model
        if results:
            best_model_name = max(results.keys(),
                                key=lambda x: results[x]['avg_metrics']['roc_auc_mean'])

            total_training_time = sum(model_training_times.values())
            logger.info(f"All models for {target_name} completed in {total_training_time:.1f}s")
            logger.info(f"Best model: {best_model_name}")

            return {
                'models': results,
                'best_model': best_model_name,
                'best_model_results': results[best_model_name],
                'total_training_time': total_training_time
            }
        else:
            raise RuntimeError(f"No models were successfully trained for {target_name}")

    def get_model_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary DataFrame of model results.

        Args:
            results: Results from train_and_evaluate

        Returns:
            DataFrame with model comparison
        """
        summary_data = []

        for model_name, model_results in results['models'].items():
            metrics = model_results['avg_metrics']
            summary_data.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{metrics['pr_auc_mean']:.3f} ± {metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}",
                'Precision': f"{metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}",
                'Recall': f"{metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}",
                'Balanced_Accuracy': f"{metrics['balanced_accuracy_mean']:.3f} ± {metrics['balanced_accuracy_std']:.3f}",
                'Training_Time': f"{model_results['training_time']:.1f}s",
                'ROC-AUC_Mean': metrics['roc_auc_mean'],
                'PR-AUC_Mean': metrics['pr_auc_mean']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC_Mean', ascending=False)

        return summary_df