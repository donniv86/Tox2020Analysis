"""
Pipeline Manager - Main orchestrator for the Tox21 modeling pipeline
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .data_preparation import Tox21DataLoader
from .feature_selector import FeatureSelector
from .model_trainer import ModelTrainer
from .result_manager import ResultManager


class PipelineManager:
    """
    Main pipeline manager that orchestrates the entire Tox21 modeling workflow.

    Responsibilities:
    - Data loading and preprocessing
    - Feature selection
    - Model training and evaluation
    - Result management and reporting
    - Progress tracking and logging
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pipeline manager.

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Initialize components
        self.data_loader = None
        self.feature_selector = None
        self.model_trainer = None
        self.result_manager = None

        # Pipeline state
        self.results = {}
        self.completed_targets = []
        self.start_time = time.time()

        # Setup logging
        self._setup_logging()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'random_state': 42,
            'cv_folds': 5,
            'feature_selection': {
                'correlation_threshold': 0.90,
                'univariate_k': 500,
                'top_n_model': 150
            },
            'models': ['RandomForest', 'LogisticRegression', 'SVM'],
            'min_samples': 100,
            'min_positive_samples': 10,
            'resume': True,
            'log_level': logging.INFO
        }

    def _setup_logging(self):
        """Setup comprehensive logging."""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/pipeline_{timestamp}.log'

        logging.basicConfig(
            level=self.config['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename

        self.logger.info("=" * 60)
        self.logger.info("TOX21 PIPELINE MANAGER INITIALIZED")
        self.logger.info("=" * 60)
        self.logger.info(f"Log file: {log_filename}")

    def log_step(self, step: str, message: str, level: str = "INFO"):
        """Log a pipeline step with timing."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"[{step}] {message} (Elapsed: {elapsed:.1f}s)")

    def load_data(self) -> bool:
        """
        Load and prepare the Tox21 dataset.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.log_step("DATA_LOADING", "Starting data loading and preparation")

            self.data_loader = Tox21DataLoader()

            # Load descriptors
            self.log_step("DATA_LOADING", "Loading molecular descriptors...")
            self.data_loader.load_descriptors()

            # Load targets
            self.log_step("DATA_LOADING", "Loading target labels...")
            self.data_loader.load_targets_from_sdf()

            # Preprocessing
            self.log_step("DATA_PREPROCESSING", "Preprocessing data...")
            self.data_loader.remove_low_variance_features(threshold=0.01)
            self.data_loader.handle_missing_values(strategy='drop')

            self.log_step("DATA_LOADING", f"Data loaded: {self.data_loader.descriptors.shape[0]} samples, "
                                        f"{self.data_loader.descriptors.shape[1]} features")

            return True

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return False

    def select_features(self, target_idx: int, target_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Perform feature selection for a target.

        Args:
            target_idx: Index of the target
            target_name: Name of the target

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        self.log_step("FEATURE_SELECTION", f"Selecting features for {target_name}")

        y = self.data_loader.targets[:, target_idx]
        X = self.data_loader.descriptors
        feature_names = self.data_loader.feature_names

        self.feature_selector = FeatureSelector(**self.config['feature_selection'])
        X_selected, selected_names = self.feature_selector.fit_transform(X, y, feature_names)

        reduction_ratio = X_selected.shape[1] / X.shape[1]
        self.log_step("FEATURE_SELECTION", f"Features reduced: {X.shape[1]} → {X_selected.shape[1]} ({reduction_ratio:.1%})")

        return X_selected, selected_names

    def train_models(self, target_idx: int, target_name: str, X_selected: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate models for a target.

        Args:
            target_idx: Index of the target
            target_name: Name of the target
            X_selected: Selected features

        Returns:
            Dictionary containing model results
        """
        self.log_step("MODEL_TRAINING", f"Training models for {target_name}")

        y = self.data_loader.targets[:, target_idx]

        self.model_trainer = ModelTrainer(
            models=self.config['models'],
            cv_folds=self.config['cv_folds'],
            random_state=self.config['random_state']
        )

        results = self.model_trainer.train_and_evaluate(X_selected, y, target_name)

        self.log_step("MODEL_TRAINING", f"Models trained for {target_name}")
        return results

    def validate_target(self, target_idx: int, target_name: str) -> bool:
        """
        Validate if a target has sufficient data for modeling.

        Args:
            target_idx: Index of the target
            target_name: Name of the target

        Returns:
            bool: True if target is valid, False otherwise
        """
        target_data = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(target_data)
        valid_data = target_data[valid_mask]

        if len(valid_data) < self.config['min_samples']:
            self.log_step("TARGET_SKIP", f"Skipping {target_name}: insufficient samples ({len(valid_data)})")
            return False

        positive_samples = np.sum(valid_data == 1)
        if positive_samples < self.config['min_positive_samples']:
            self.log_step("TARGET_SKIP", f"Skipping {target_name}: too few positive samples ({positive_samples})")
            return False

        return True

    def process_target(self, target_idx: int, target_name: str) -> bool:
        """
        Process a single target through the complete pipeline.

        Args:
            target_idx: Index of the target
            target_name: Name of the target

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate target
            if not self.validate_target(target_idx, target_name):
                return False

            # Feature selection
            X_selected, selected_names = self.select_features(target_idx, target_name)

            # Model training
            results = self.train_models(target_idx, target_name, X_selected)

            # Save results
            self.result_manager = ResultManager()
            self.result_manager.save_target_results(
                target_name, results, self.feature_selector, selected_names
            )

            # Update state
            self.results[target_name] = results
            self.completed_targets.append(target_name)

            self.log_step("TARGET_COMPLETE", f"✓ Completed {target_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {target_name}: {str(e)}")
            return False

    def run_pipeline(self, target_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            target_indices: List of target indices to process. If None, process all targets.

        Returns:
            Dictionary containing pipeline results
        """
        self.log_step("PIPELINE_START", "Starting Tox21 modeling pipeline")

        # Load data
        if not self.load_data():
            raise RuntimeError("Data loading failed")

        # Determine targets to process
        if target_indices is None:
            target_indices = range(len(self.data_loader.target_names))

        # Load progress if resuming
        if self.config['resume']:
            self._load_progress()

        total_targets = len(target_indices)
        self.log_step("PIPELINE_PROGRESS", f"Processing {total_targets} targets")

        # Process each target
        for i, target_idx in enumerate(target_indices):
            target_name = self.data_loader.target_names[target_idx]

            # Skip if already completed
            if self.config['resume'] and target_name in self.completed_targets:
                self.log_step("PIPELINE_PROGRESS", f"Skipping {target_name} (already completed)")
                continue

            self.log_step("TARGET_PROCESSING", f"Processing target {i + 1}/{total_targets}: {target_name}")
            self.process_target(target_idx, target_name)

            # Save progress after each target
            if self.config['resume']:
                self._save_progress()

        # Create summary report
        self._create_summary_report()

        total_time = time.time() - self.start_time
        self.log_step("PIPELINE_COMPLETE", f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")

        return self.results

    def _save_progress(self):
        """Save current progress to file."""
        progress_data = {
            'completed_targets': self.completed_targets,
            'timestamp': datetime.now().isoformat()
        }

        with open('logs/pipeline_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)

    def _load_progress(self):
        """Load progress from file if exists."""
        progress_file = 'logs/pipeline_progress.json'
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)

            self.completed_targets = progress_data.get('completed_targets', [])
            self.log_step("PROGRESS_LOAD", f"Loaded progress: {len(self.completed_targets)} targets completed")

    def _create_summary_report(self):
        """Create a summary report of all results."""
        if not self.results:
            return

        self.log_step("SUMMARY_REPORT", "Creating summary report")

        summary_data = []
        for target_name, target_results in self.results.items():
            best_model_name = target_results['best_model']
            best_metrics = target_results['models'][best_model_name]['avg_metrics']

            summary_data.append({
                'Target': target_name,
                'Best_Model': best_model_name,
                'ROC-AUC': f"{best_metrics['roc_auc_mean']:.3f} ± {best_metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{best_metrics['pr_auc_mean']:.3f} ± {best_metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{best_metrics['f1_mean']:.3f} ± {best_metrics['f1_std']:.3f}",
                'ROC-AUC_Mean': best_metrics['roc_auc_mean'],
                'PR-AUC_Mean': best_metrics['pr_auc_mean']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC_Mean', ascending=False)

        os.makedirs('results', exist_ok=True)
        summary_df.to_csv('results/pipeline_summary.csv', index=False)

        avg_roc_auc = summary_df['ROC-AUC_Mean'].mean()
        self.log_step("SUMMARY_STATS", f"Average ROC-AUC: {avg_roc_auc:.3f}")