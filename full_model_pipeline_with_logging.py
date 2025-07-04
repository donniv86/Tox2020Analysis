#!/usr/bin/env python3
"""
Full-Fledged Model Building Pipeline with Comprehensive Logging

This script provides a complete workflow with:
1. Comprehensive logging and progress tracking
2. Timestamp tracking for each step
3. Ability to resume from interruption
4. Detailed progress reporting
5. Performance monitoring
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import warnings
import logging
import time
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from data_preparation import Tox21DataLoader
from feature_selector import FeatureSelector

class FullModelPipelineWithLogging:
    """
    Complete model building pipeline with comprehensive logging
    """

    def __init__(self, random_state=42, log_level=logging.INFO):
        self.random_state = random_state
        self.results = {}
        self.feature_selectors = {}
        self.best_models = {}
        self.start_time = time.time()
        self.pipeline_log = []

        # Setup logging
        self.setup_logging(log_level)

        # Load progress if exists
        self.load_progress()

    def setup_logging(self, log_level):
        """Setup comprehensive logging"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)

        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/full_pipeline_{timestamp}.log'

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename

        self.logger.info("=" * 80)
        self.logger.info("FULL MODEL PIPELINE WITH LOGGING INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Log file: {log_filename}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_step(self, step_name, message, level="INFO"):
        """Log a step with timestamp and duration tracking"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'message': message,
            'elapsed_seconds': elapsed
        }

        self.pipeline_log.append(log_entry)

        if level == "INFO":
            self.logger.info(f"[{step_name}] {message} (Elapsed: {elapsed:.1f}s)")
        elif level == "WARNING":
            self.logger.warning(f"[{step_name}] {message} (Elapsed: {elapsed:.1f}s)")
        elif level == "ERROR":
            self.logger.error(f"[{step_name}] {message} (Elapsed: {elapsed:.1f}s)")

    def save_progress(self):
        """Save current progress to file"""
        progress_data = {
            'completed_targets': list(self.results.keys()),
            'pipeline_log': self.pipeline_log,
            'timestamp': datetime.now().isoformat()
        }

        with open('logs/pipeline_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)

        self.logger.info("Progress saved to logs/pipeline_progress.json")

    def load_progress(self):
        """Load progress from file if exists"""
        progress_file = 'logs/pipeline_progress.json'
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)

            self.logger.info(f"Found previous progress: {len(progress_data.get('completed_targets', []))} targets completed")
            self.logger.info(f"Completed targets: {progress_data.get('completed_targets', [])}")

            # Load pipeline log
            self.pipeline_log = progress_data.get('pipeline_log', [])

            return progress_data.get('completed_targets', [])
        return []

    def load_and_prepare_data(self):
        """Load and prepare the Tox21 dataset with logging"""
        self.log_step("DATA_LOADING", "Starting data loading and preparation")

        # Initialize data loader
        loader = Tox21DataLoader()

        # Load descriptors
        self.log_step("DATA_LOADING", "Loading descriptors...")
        loader.load_descriptors()
        self.log_step("DATA_LOADING", f"Loaded: {loader.descriptors.shape[0]} samples, {loader.descriptors.shape[1]} features")

        # Load targets
        self.log_step("DATA_LOADING", "Loading target labels...")
        loader.load_targets_from_sdf()
        self.log_step("DATA_LOADING", f"Loaded: {loader.targets.shape[1]} targets")

        # Preprocessing
        self.log_step("DATA_PREPROCESSING", "Starting data preprocessing...")
        loader.remove_low_variance_features(threshold=0.01)
        loader.handle_missing_values(strategy='drop')

        self.log_step("DATA_PREPROCESSING", f"Final dataset: {loader.descriptors.shape[0]} samples, {loader.descriptors.shape[1]} features")

        # Get target statistics
        target_stats = loader.get_target_statistics()
        self.log_step("DATA_ANALYSIS", f"Target statistics calculated for {len(target_stats)} targets")

        self.loader = loader
        self.target_stats = target_stats

        return loader, target_stats

    def perform_feature_selection(self, target_idx, target_name, feature_params=None):
        """Perform feature selection for a specific target with logging"""
        self.log_step("FEATURE_SELECTION", f"Starting feature selection for {target_name}")

        # Default feature selection parameters
        if feature_params is None:
            feature_params = {
                'correlation_threshold': 0.90,
                'univariate_k': 500,
                'top_n_model': 150
            }

        self.log_step("FEATURE_SELECTION", f"Parameters: {feature_params}")

        # Get target data
        y = self.loader.targets[:, target_idx]
        X = self.loader.descriptors
        feature_names = self.loader.feature_names

        # Perform feature selection
        start_time = time.time()
        selector = FeatureSelector(**feature_params)
        X_selected, selected_names = selector.fit_transform(X, y, feature_names)
        selection_time = time.time() - start_time

        reduction_ratio = X_selected.shape[1] / X.shape[1]
        self.log_step("FEATURE_SELECTION", f"Features reduced: {X.shape[1]} → {X_selected.shape[1]} ({reduction_ratio:.1%}) in {selection_time:.1f}s")

        # Store selector for later use
        self.feature_selectors[target_name] = {
            'selector': selector,
            'selected_features': selected_names,
            'X_selected': X_selected,
            'selection_time': selection_time
        }

        return X_selected, selected_names, selector

    def train_and_evaluate_models(self, target_idx, target_name, cv_folds=5):
        """Train and evaluate models for a specific target with detailed logging"""
        self.log_step("MODEL_TRAINING", f"Starting model training for {target_name} with {cv_folds}-fold CV")

        # Get selected features
        X_selected = self.feature_selectors[target_name]['X_selected']
        y = self.loader.targets[:, target_idx]

        # Define models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state,
                n_jobs=-1, class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state,
                class_weight='balanced'
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True,
                random_state=self.random_state, class_weight='balanced'
            )
        }

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Store results for each model
        target_results = {}
        model_training_times = {}

        for model_name, model in models.items():
            self.log_step("MODEL_TRAINING", f"Training {model_name} for {target_name}")
            model_start_time = time.time()

            # Initialize results storage
            fold_results = {
                'roc_auc': [], 'pr_auc': [], 'f1': [],
                'precision': [], 'recall': [], 'accuracy': [], 'balanced_accuracy': []
            }

            # Cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y)):
                fold_start_time = time.time()

                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)

                # Store results
                for metric, value in metrics.items():
                    if metric in fold_results:
                        fold_results[metric].append(value)

                fold_time = time.time() - fold_start_time
                self.log_step("CV_FOLD", f"{model_name} - Fold {fold_idx + 1}/{cv_folds} completed in {fold_time:.1f}s")

            # Calculate average metrics
            avg_metrics = {}
            for metric, values in fold_results.items():
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)

            model_time = time.time() - model_start_time
            model_training_times[model_name] = model_time

            target_results[model_name] = {
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,
                'model': model,
                'training_time': model_time
            }

            self.log_step("MODEL_TRAINING", f"{model_name} completed in {model_time:.1f}s")
            self.log_step("MODEL_PERFORMANCE", f"{model_name} - ROC-AUC: {avg_metrics['roc_auc_mean']:.3f}±{avg_metrics['roc_auc_std']:.3f}")

        # Find best model
        best_model_name = max(target_results.keys(),
                            key=lambda x: target_results[x]['avg_metrics']['roc_auc_mean'])

        self.best_models[target_name] = {
            'model_name': best_model_name,
            'model': target_results[best_model_name]['model'],
            'metrics': target_results[best_model_name]['avg_metrics'],
            'training_time': target_results[best_model_name]['training_time']
        }

        self.results[target_name] = target_results

        total_training_time = sum(model_training_times.values())
        self.log_step("MODEL_TRAINING", f"All models for {target_name} completed in {total_training_time:.1f}s")
        self.log_step("MODEL_SELECTION", f"Best model for {target_name}: {best_model_name}")

        return target_results

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
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

    def save_results(self, target_name, save_dir='results'):
        """Save all results for a target with logging"""
        self.log_step("RESULT_SAVING", f"Saving results for {target_name}")

        import pickle

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Save detailed results
        results_path = os.path.join(save_dir, f'{target_name}_full_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results[target_name], f)

        # Save feature selector
        selector_path = os.path.join(save_dir, f'{target_name}_feature_selector.pkl')
        with open(selector_path, 'wb') as f:
            pickle.dump(self.feature_selectors[target_name], f)

        # Save best model
        best_model_path = os.path.join(save_dir, f'{target_name}_best_model.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_models[target_name], f)

        # Create and save comparison table
        comparison_df = self.create_comprehensive_report(target_name)
        csv_path = os.path.join(save_dir, f'{target_name}_model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)

        self.log_step("RESULT_SAVING", f"Results saved: {results_path}, {selector_path}, {best_model_path}, {csv_path}")

    def create_comprehensive_report(self, target_name):
        """Create comprehensive report for a target"""
        self.log_step("REPORTING", f"Creating comprehensive report for {target_name}")

        # Get results
        target_results = self.results[target_name]

        # Create comparison table
        comparison_data = []
        for model_name, results in target_results.items():
            metrics = results['avg_metrics']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{metrics['pr_auc_mean']:.3f} ± {metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}",
                'Precision': f"{metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}",
                'Recall': f"{metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}",
                'Balanced_Accuracy': f"{metrics['balanced_accuracy_mean']:.3f} ± {metrics['balanced_accuracy_std']:.3f}",
                'Training_Time': f"{results['training_time']:.1f}s",
                'ROC-AUC_Mean': metrics['roc_auc_mean'],
                'PR-AUC_Mean': metrics['pr_auc_mean']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC_Mean', ascending=False)

        self.log_step("REPORTING", f"Model comparison for {target_name} completed")

        return comparison_df

    def run_full_pipeline(self, target_indices=None, cv_folds=5, resume=True):
        """Run the complete pipeline with comprehensive logging"""
        self.log_step("PIPELINE_START", "Starting full-fledged model building pipeline")

        # Step 1: Data preparation
        self.log_step("PIPELINE_START", "Step 1: Data loading and preparation")
        loader, target_stats = self.load_and_prepare_data()

        # Step 2: Model building for each target
        self.log_step("PIPELINE_START", "Step 2: Model building")

        # Select targets to process
        if target_indices is None:
            target_indices = range(len(loader.target_names))

        # Check for completed targets if resuming
        completed_targets = []
        if resume:
            completed_targets = self.load_progress()

        total_targets = len(target_indices)
        completed_count = len(completed_targets)

        self.log_step("PIPELINE_PROGRESS", f"Processing {total_targets} targets, {completed_count} already completed")

        for i, target_idx in enumerate(target_indices):
            target_name = loader.target_names[target_idx]

            # Skip if already completed
            if resume and target_name in completed_targets:
                self.log_step("PIPELINE_PROGRESS", f"Skipping {target_name} (already completed)")
                continue

            self.log_step("TARGET_PROCESSING", f"Processing target {i + 1}/{total_targets}: {target_name}")

            try:
                # Skip targets with too few samples
                target_data = loader.targets[:, target_idx]
                valid_mask = ~np.isnan(target_data)
                valid_data = target_data[valid_mask]

                if len(valid_data) < 100:
                    self.log_step("TARGET_SKIP", f"Skipping {target_name}: insufficient samples ({len(valid_data)})")
                    continue

                if np.sum(valid_data == 1) < 10:
                    self.log_step("TARGET_SKIP", f"Skipping {target_name}: too few positive samples ({np.sum(valid_data == 1)})")
                    continue

                # Feature selection
                X_selected, selected_names, selector = self.perform_feature_selection(target_idx, target_name)

                # Model training and evaluation
                target_results = self.train_and_evaluate_models(target_idx, target_name, cv_folds)

                # Create report and save results
                self.save_results(target_name)

                # Save progress after each target
                self.save_progress()

                self.log_step("TARGET_COMPLETE", f"✓ Completed {target_name}")

            except Exception as e:
                self.log_step("TARGET_ERROR", f"Error processing {target_name}: {str(e)}", level="ERROR")
                continue

        # Step 3: Summary report
        self.log_step("PIPELINE_START", "Step 3: Creating summary report")
        self.create_summary_report()

        total_time = time.time() - self.start_time
        self.log_step("PIPELINE_COMPLETE", f"Full pipeline completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")

        return self.results

    def create_summary_report(self):
        """Create a summary report of all targets"""
        self.log_step("SUMMARY_REPORT", "Creating summary report")

        summary_data = []
        for target_name in self.results.keys():
            best_model_info = self.best_models[target_name]
            metrics = best_model_info['metrics']

            summary_data.append({
                'Target': target_name,
                'Best_Model': best_model_info['model_name'],
                'ROC-AUC': f"{metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{metrics['pr_auc_mean']:.3f} ± {metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}",
                'Training_Time': f"{best_model_info['training_time']:.1f}s",
                'ROC-AUC_Mean': metrics['roc_auc_mean'],
                'PR-AUC_Mean': metrics['pr_auc_mean']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC_Mean', ascending=False)

        # Save summary
        summary_df.to_csv('results/full_pipeline_summary.csv', index=False)
        self.log_step("SUMMARY_REPORT", "Summary saved to results/full_pipeline_summary.csv")

        # Log summary statistics
        avg_roc_auc = summary_df['ROC-AUC_Mean'].mean()
        avg_pr_auc = summary_df['PR-AUC_Mean'].mean()
        total_training_time = sum([float(x.replace('s', '')) for x in summary_df['Training_Time']])

        self.log_step("SUMMARY_STATS", f"Average ROC-AUC: {avg_roc_auc:.3f}")
        self.log_step("SUMMARY_STATS", f"Average PR-AUC: {avg_pr_auc:.3f}")
        self.log_step("SUMMARY_STATS", f"Total training time: {total_training_time:.1f}s")

        return summary_df


def main():
    """Run the full model building pipeline with logging"""

    # Initialize pipeline with logging
    pipeline = FullModelPipelineWithLogging(random_state=42, log_level=logging.INFO)

    # Run pipeline for a few targets (to avoid long runtime)
    # You can change this to process all targets or specific ones
    target_indices = [0, 7, 11]  # NR-AR, SR-ARE, SR-p53

    try:
        results = pipeline.run_full_pipeline(target_indices=target_indices, cv_folds=5, resume=True)
        print(f"\nPipeline completed successfully! Check logs/full_pipeline_*.log for detailed logs.")
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user. Progress saved. Resume later with resume=True.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        print("Check logs for detailed error information.")


if __name__ == "__main__":
    main()