"""
Utility functions for the Tox21 modeling pipeline.

This module provides common operations for working with trained models and results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from src.result_manager import ResultManager
from config import PIPELINE_CONFIG

class Tox21Utils:
    """Utility class for common Tox21 operations."""

    def __init__(self, results_dir: str = 'results'):
        """
        Initialize the utility class.

        Args:
            results_dir: Directory containing results
        """
        self.result_manager = ResultManager(results_dir)

    def list_available_targets(self) -> List[str]:
        """Get list of targets with trained models."""
        return self.result_manager.get_available_targets()

    def load_model_and_features(self, target_name: str) -> Dict[str, Any]:
        """
        Load the best model and feature selector for a target.

        Args:
            target_name: Name of the target

        Returns:
            Dictionary containing model, feature selector, and selected features
        """
        results = self.result_manager.load_target_results(target_name)

        return {
            'model': results['best_model'],
            'feature_selector': results['feature_selector'],
            'selected_features': results['selected_features'],
            'training_results': results['training_results']
        }

    def predict_toxicity(self, target_name: str, descriptors: np.ndarray,
                        feature_names: List[str]) -> Dict[str, Any]:
        """
        Make toxicity predictions for new compounds.

        Args:
            target_name: Name of the target
            descriptors: Molecular descriptors (n_samples, n_features)
            feature_names: Names of the descriptors

        Returns:
            Dictionary containing predictions and probabilities
        """
        # Load model and feature selector
        model_data = self.load_model_and_features(target_name)
        model = model_data['model']
        feature_selector = model_data['feature_selector']

        # Select features
        X_selected, _ = feature_selector.transform(descriptors, feature_names)

        # Make predictions
        predictions = model.predict(X_selected)
        probabilities = model.predict_proba(X_selected)[:, 1]

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'target_name': target_name
        }

    def get_performance_summary(self, target_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a target.

        Args:
            target_name: Name of the target

        Returns:
            Dictionary containing performance metrics
        """
        results = self.result_manager.load_target_results(target_name)
        training_results = results['training_results']

        best_model_name = training_results['best_model']
        best_metrics = training_results['models'][best_model_name]['avg_metrics']

        return {
            'target_name': target_name,
            'best_model': best_model_name,
            'roc_auc': best_metrics['roc_auc_mean'],
            'pr_auc': best_metrics['pr_auc_mean'],
            'f1_score': best_metrics['f1_mean'],
            'precision': best_metrics['precision_mean'],
            'recall': best_metrics['recall_mean'],
            'balanced_accuracy': best_metrics['balanced_accuracy_mean'],
            'training_time': training_results['total_training_time']
        }

    def compare_targets(self, target_names: List[str]) -> pd.DataFrame:
        """
        Compare performance across multiple targets.

        Args:
            target_names: List of target names to compare

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for target_name in target_names:
            try:
                summary = self.get_performance_summary(target_name)
                comparison_data.append(summary)
            except Exception as e:
                print(f"Error loading {target_name}: {str(e)}")
                continue

        if not comparison_data:
            raise ValueError("No valid targets found for comparison")

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('roc_auc', ascending=False)

        return df

    def export_model_info(self, target_name: str, output_file: str = None) -> Dict[str, Any]:
        """
        Export comprehensive model information.

        Args:
            target_name: Name of the target
            output_file: Optional output file path

        Returns:
            Dictionary containing model information
        """
        results = self.result_manager.load_target_results(target_name)
        training_results = results['training_results']

        # Get performance summary
        performance = self.get_performance_summary(target_name)

        # Get model comparison
        from src.model_trainer import ModelTrainer
        trainer = ModelTrainer(models=[])
        model_comparison = trainer.get_model_summary(training_results)

        model_info = {
            'target_name': target_name,
            'performance': performance,
            'model_comparison': model_comparison.to_dict('records'),
            'selected_features_count': len(results['selected_features']),
            'selected_features': results['selected_features'][:10],  # First 10 features
            'training_config': {
                'cv_folds': PIPELINE_CONFIG['cv_folds'],
                'models_trained': list(training_results['models'].keys()),
                'total_training_time': training_results['total_training_time']
            }
        }

        # Save to file if specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)

        return model_info

    def create_prediction_report(self, target_name: str, predictions: np.ndarray,
                                probabilities: np.ndarray, compound_ids: List[str] = None) -> pd.DataFrame:
        """
        Create a prediction report DataFrame.

        Args:
            target_name: Name of the target
            predictions: Binary predictions
            probabilities: Prediction probabilities
            compound_ids: Optional list of compound identifiers

        Returns:
            DataFrame with prediction results
        """
        if compound_ids is None:
            compound_ids = [f"Compound_{i+1}" for i in range(len(predictions))]

        report_data = {
            'Compound_ID': compound_ids,
            'Target': target_name,
            'Prediction': predictions,
            'Probability': probabilities,
            'Toxicity_Class': ['Toxic' if p == 1 else 'Non-toxic' for p in predictions],
            'Confidence': np.abs(probabilities - 0.5) * 2  # Distance from 0.5
        }

        df = pd.DataFrame(report_data)
        df = df.sort_values('Probability', ascending=False)

        return df


def main():
    """Example usage of the utility functions."""
    utils = Tox21Utils()

    # List available targets
    targets = utils.list_available_targets()
    print(f"Available targets: {targets}")

    if not targets:
        print("No trained models found. Run the pipeline first.")
        return

    # Get performance summary for first target
    target_name = targets[0]
    summary = utils.get_performance_summary(target_name)
    print(f"\nPerformance summary for {target_name}:")
    for key, value in summary.items():
        if key != 'target_name':
            print(f"  {key}: {value}")

    # Compare all targets
    print(f"\nComparing all targets:")
    comparison = utils.compare_targets(targets)
    print(comparison[['target_name', 'best_model', 'roc_auc', 'f1_score']].to_string(index=False))


if __name__ == "__main__":
    main()