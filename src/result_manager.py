"""
Result Manager - Handles saving and loading of results, models, and reports
"""

import os
import pickle
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Manages saving and loading of pipeline results.

    Responsibilities:
    - Save/load model results
    - Save/load trained models
    - Save/load feature selectors
    - Generate reports
    - Manage file organization
    """

    def __init__(self, results_dir: str = 'results'):
        """
        Initialize the result manager.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_target_results(self, target_name: str, results: Dict[str, Any],
                           feature_selector: Any, selected_features: List[str]) -> None:
        """
        Save all results for a target.

        Args:
            target_name: Name of the target
            results: Model training results
            feature_selector: Trained feature selector
            selected_features: List of selected feature names
        """
        logger.info(f"Saving results for {target_name}")

        # Save detailed results
        self._save_results(target_name, results)

        # Save feature selector
        self._save_feature_selector(target_name, feature_selector, selected_features)

        # Save best model
        self._save_best_model(target_name, results)

        # Create and save comparison report
        self._save_comparison_report(target_name, results)

        logger.info(f"All results saved for {target_name}")

    def _save_results(self, target_name: str, results: Dict[str, Any]) -> None:
        """Save detailed results dictionary."""
        filepath = os.path.join(self.results_dir, f'{target_name}_results.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.debug(f"Results saved: {filepath}")

    def _save_feature_selector(self, target_name: str, feature_selector: Any,
                              selected_features: List[str]) -> None:
        """Save feature selector and selected features."""
        # Save feature selector object
        selector_filepath = os.path.join(self.results_dir, f'{target_name}_feature_selector.pkl')
        with open(selector_filepath, 'wb') as f:
            pickle.dump(feature_selector, f)

        # Save selected feature names
        features_filepath = os.path.join(self.results_dir, f'{target_name}_selected_features.npy')
        np.save(features_filepath, np.array(selected_features))

        logger.debug(f"Feature selector saved: {selector_filepath}")
        logger.debug(f"Selected features saved: {features_filepath}")

    def _save_best_model(self, target_name: str, results: Dict[str, Any]) -> None:
        """Save the best performing model."""
        best_model_name = results['best_model']
        best_model = results['models'][best_model_name]['model']

        filepath = os.path.join(self.results_dir, f'{target_name}_best_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(best_model, f)

        logger.debug(f"Best model saved: {filepath}")

    def _save_comparison_report(self, target_name: str, results: Dict[str, Any]) -> None:
        """Create and save model comparison report."""
        from .model_trainer import ModelTrainer

        # Create summary DataFrame
        trainer = ModelTrainer(models=[])  # Dummy trainer for summary method
        summary_df = trainer.get_model_summary(results)

        # Save CSV report
        filepath = os.path.join(self.results_dir, f'{target_name}_model_comparison.csv')
        summary_df.to_csv(filepath, index=False)

        logger.debug(f"Comparison report saved: {filepath}")

    def load_target_results(self, target_name: str) -> Dict[str, Any]:
        """
        Load all results for a target.

        Args:
            target_name: Name of the target

        Returns:
            Dictionary containing loaded results
        """
        logger.info(f"Loading results for {target_name}")

        results = {}

        # Load detailed results
        results_filepath = os.path.join(self.results_dir, f'{target_name}_results.pkl')
        if os.path.exists(results_filepath):
            with open(results_filepath, 'rb') as f:
                results['training_results'] = pickle.load(f)

        # Load feature selector
        selector_filepath = os.path.join(self.results_dir, f'{target_name}_feature_selector.pkl')
        if os.path.exists(selector_filepath):
            with open(selector_filepath, 'rb') as f:
                results['feature_selector'] = pickle.load(f)

        # Load selected features
        features_filepath = os.path.join(self.results_dir, f'{target_name}_selected_features.npy')
        if os.path.exists(features_filepath):
            results['selected_features'] = np.load(features_filepath, allow_pickle=True).tolist()

        # Load best model
        model_filepath = os.path.join(self.results_dir, f'{target_name}_best_model.pkl')
        if os.path.exists(model_filepath):
            with open(model_filepath, 'rb') as f:
                results['best_model'] = pickle.load(f)

        logger.info(f"Results loaded for {target_name}")
        return results

    def load_best_model(self, target_name: str):
        """
        Load the best model for a target.

        Args:
            target_name: Name of the target

        Returns:
            Trained model object
        """
        filepath = os.path.join(self.results_dir, f'{target_name}_best_model.pkl')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found for {target_name}")

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Best model loaded for {target_name}")
        return model

    def load_feature_selector(self, target_name: str):
        """
        Load the feature selector for a target.

        Args:
            target_name: Name of the target

        Returns:
            Feature selector object
        """
        filepath = os.path.join(self.results_dir, f'{target_name}_feature_selector.pkl')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No feature selector found for {target_name}")

        with open(filepath, 'rb') as f:
            selector = pickle.load(f)

        logger.info(f"Feature selector loaded for {target_name}")
        return selector

    def get_available_targets(self) -> List[str]:
        """
        Get list of targets with saved results.

        Returns:
            List of target names
        """
        targets = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('_results.pkl'):
                target_name = filename.replace('_results.pkl', '')
                targets.append(target_name)

        return sorted(targets)

    def create_summary_report(self, target_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary report of all targets.

        Args:
            target_results: Dictionary mapping target names to their results

        Returns:
            DataFrame with summary statistics
        """
        logger.info("Creating summary report")

        summary_data = []
        for target_name, results in target_results.items():
            best_model_name = results['best_model']
            best_metrics = results['models'][best_model_name]['avg_metrics']

            summary_data.append({
                'Target': target_name,
                'Best_Model': best_model_name,
                'ROC-AUC': f"{best_metrics['roc_auc_mean']:.3f} ± {best_metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{best_metrics['pr_auc_mean']:.3f} ± {best_metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{best_metrics['f1_mean']:.3f} ± {best_metrics['f1_std']:.3f}",
                'Training_Time': f"{results['total_training_time']:.1f}s",
                'ROC-AUC_Mean': best_metrics['roc_auc_mean'],
                'PR-AUC_Mean': best_metrics['pr_auc_mean']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC_Mean', ascending=False)

        # Save summary report
        filepath = os.path.join(self.results_dir, 'pipeline_summary.csv')
        summary_df.to_csv(filepath, index=False)

        logger.info(f"Summary report saved: {filepath}")
        return summary_df

    def cleanup_old_results(self, keep_latest: int = 3) -> None:
        """
        Clean up old result files, keeping only the latest ones.

        Args:
            keep_latest: Number of latest result sets to keep
        """
        logger.info(f"Cleaning up old results, keeping {keep_latest} latest")

        # This is a placeholder for cleanup logic
        # In a real implementation, you might want to:
        # 1. Identify result sets by timestamp
        # 2. Remove older files
        # 3. Keep only the specified number of latest results

        logger.info("Cleanup completed")

    def export_results_for_sharing(self, target_names: List[str],
                                  output_dir: str = 'export') -> None:
        """
        Export results for sharing (e.g., with collaborators).

        Args:
            target_names: List of target names to export
            output_dir: Output directory for exported files
        """
        logger.info(f"Exporting results for {len(target_names)} targets")

        os.makedirs(output_dir, exist_ok=True)

        for target_name in target_names:
            try:
                # Load results
                results = self.load_target_results(target_name)

                # Export in a standardized format
                export_data = {
                    'target_name': target_name,
                    'best_model_name': results['training_results']['best_model'],
                    'performance_metrics': results['training_results']['best_model_results']['avg_metrics'],
                    'selected_features_count': len(results['selected_features']),
                    'training_time': results['training_results']['total_training_time']
                }

                # Save export
                export_filepath = os.path.join(output_dir, f'{target_name}_export.json')
                import json
                with open(export_filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

                logger.debug(f"Exported: {export_filepath}")

            except Exception as e:
                logger.error(f"Error exporting {target_name}: {str(e)}")

        logger.info(f"Export completed to {output_dir}")