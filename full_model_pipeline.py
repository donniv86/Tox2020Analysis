#!/usr/bin/env python3
"""
Full-Fledged Model Building Pipeline for Tox21 Dataset

This script provides a complete workflow:
1. Data loading and preprocessing
2. Feature selection for each target
3. Cross-validation model training
4. Comprehensive evaluation
5. Result analysis and saving
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
warnings.filterwarnings('ignore')

from data_preparation import Tox21DataLoader
from feature_selector import FeatureSelector

class FullModelPipeline:
    """
    Complete model building pipeline for Tox21 toxicity prediction
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.feature_selectors = {}
        self.best_models = {}

    def load_and_prepare_data(self):
        """Load and prepare the Tox21 dataset"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 80)

        # Initialize data loader
        loader = Tox21DataLoader()

        # Load descriptors
        print("\n1.1 Loading descriptors...")
        loader.load_descriptors()
        print(f"   Loaded: {loader.descriptors.shape[0]} samples, {loader.descriptors.shape[1]} features")

        # Load targets
        print("\n1.2 Loading target labels...")
        loader.load_targets_from_sdf()
        print(f"   Loaded: {loader.targets.shape[1]} targets")

        # Preprocessing
        print("\n1.3 Data preprocessing...")
        loader.remove_low_variance_features(threshold=0.01)
        loader.handle_missing_values(strategy='drop')

        print(f"   Final dataset: {loader.descriptors.shape[0]} samples, {loader.descriptors.shape[1]} features")

        # Get target statistics
        target_stats = loader.get_target_statistics()
        print(f"\n1.4 Target statistics:")
        print(target_stats.to_string(index=False))

        self.loader = loader
        self.target_stats = target_stats

        return loader, target_stats

    def perform_feature_selection(self, target_idx, target_name, feature_params=None):
        """Perform feature selection for a specific target"""
        print(f"\n2.2 Feature selection for {target_name}...")

        # Default feature selection parameters
        if feature_params is None:
            feature_params = {
                'correlation_threshold': 0.90,
                'univariate_k': 500,
                'top_n_model': 150
            }

        # Get target data
        y = self.loader.targets[:, target_idx]
        X = self.loader.descriptors
        feature_names = self.loader.feature_names

        # Perform feature selection
        selector = FeatureSelector(**feature_params)
        X_selected, selected_names = selector.fit_transform(X, y, feature_names)

        print(f"   Features reduced: {X.shape[1]} → {X_selected.shape[1]} ({X_selected.shape[1]/X.shape[1]:.1%})")

        # Store selector for later use
        self.feature_selectors[target_name] = {
            'selector': selector,
            'selected_features': selected_names,
            'X_selected': X_selected
        }

        return X_selected, selected_names, selector

    def train_and_evaluate_models(self, target_idx, target_name, cv_folds=5):
        """Train and evaluate models for a specific target"""
        print(f"\n2.3 Model training and evaluation for {target_name}...")

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

        for model_name, model in models.items():
            print(f"\n   Training {model_name}...")

            # Initialize results storage
            fold_results = {
                'roc_auc': [], 'pr_auc': [], 'f1': [],
                'precision': [], 'recall': [], 'accuracy': [], 'balanced_accuracy': []
            }

            # Cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y)):
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

            # Calculate average metrics
            avg_metrics = {}
            for metric, values in fold_results.items():
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)

            target_results[model_name] = {
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,
                'model': model
            }

            print(f"     ROC-AUC: {avg_metrics['roc_auc_mean']:.3f} ± {avg_metrics['roc_auc_std']:.3f}")
            print(f"     PR-AUC:  {avg_metrics['pr_auc_mean']:.3f} ± {avg_metrics['pr_auc_std']:.3f}")
            print(f"     F1:      {avg_metrics['f1_mean']:.3f} ± {avg_metrics['f1_std']:.3f}")

        # Find best model
        best_model_name = max(target_results.keys(),
                            key=lambda x: target_results[x]['avg_metrics']['roc_auc_mean'])

        self.best_models[target_name] = {
            'model_name': best_model_name,
            'model': target_results[best_model_name]['model'],
            'metrics': target_results[best_model_name]['avg_metrics']
        }

        self.results[target_name] = target_results

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

    def create_comprehensive_report(self, target_name):
        """Create comprehensive report for a target"""
        print(f"\n2.4 Creating comprehensive report for {target_name}...")

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
                'ROC-AUC_Mean': metrics['roc_auc_mean'],
                'PR-AUC_Mean': metrics['pr_auc_mean']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC_Mean', ascending=False)

        print(f"\n   Model Comparison for {target_name}:")
        print(comparison_df[['Model', 'ROC-AUC', 'PR-AUC', 'F1-Score', 'Balanced_Accuracy']].to_string(index=False))

        return comparison_df

    def save_results(self, target_name, save_dir='results'):
        """Save all results for a target"""
        import os
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

        # Save comparison table
        comparison_df = self.create_comprehensive_report(target_name)
        csv_path = os.path.join(save_dir, f'{target_name}_model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)

        print(f"\n   Results saved to {save_dir}/")
        print(f"     - Detailed results: {results_path}")
        print(f"     - Feature selector: {selector_path}")
        print(f"     - Best model: {best_model_path}")
        print(f"     - Comparison table: {csv_path}")

    def run_full_pipeline(self, target_indices=None, cv_folds=5):
        """Run the complete pipeline for specified targets"""
        print("=" * 80)
        print("FULL-FLEDGED MODEL BUILDING PIPELINE")
        print("=" * 80)

        # Step 1: Data preparation
        loader, target_stats = self.load_and_prepare_data()

        # Step 2: Model building for each target
        print("\n" + "=" * 80)
        print("STEP 2: MODEL BUILDING")
        print("=" * 80)

        # Select targets to process
        if target_indices is None:
            # Process all targets
            target_indices = range(len(loader.target_names))

        for target_idx in target_indices:
            target_name = loader.target_names[target_idx]

            print(f"\n2.1 Processing target {target_idx + 1}/{len(target_indices)}: {target_name}")

            # Skip targets with too few samples
            target_data = loader.targets[:, target_idx]
            valid_mask = ~np.isnan(target_data)
            valid_data = target_data[valid_mask]

            if len(valid_data) < 100:
                print(f"   Skipping {target_name}: insufficient samples ({len(valid_data)})")
                continue

            if np.sum(valid_data == 1) < 10:
                print(f"   Skipping {target_name}: too few positive samples ({np.sum(valid_data == 1)})")
                continue

            # Feature selection
            X_selected, selected_names, selector = self.perform_feature_selection(target_idx, target_name)

            # Model training and evaluation
            target_results = self.train_and_evaluate_models(target_idx, target_name, cv_folds)

            # Create report and save results
            self.save_results(target_name)

            print(f"   ✓ Completed {target_name}")

        # Step 3: Summary report
        print("\n" + "=" * 80)
        print("STEP 3: SUMMARY REPORT")
        print("=" * 80)

        self.create_summary_report()

        print("\n" + "=" * 80)
        print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    def create_summary_report(self):
        """Create a summary report of all targets"""
        print("\n3.1 Creating summary report...")

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
                'ROC-AUC_Mean': metrics['roc_auc_mean'],
                'PR-AUC_Mean': metrics['pr_auc_mean']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC_Mean', ascending=False)

        print(f"\n   Summary of all targets:")
        print(summary_df[['Target', 'Best_Model', 'ROC-AUC', 'PR-AUC', 'F1-Score']].to_string(index=False))

        # Save summary
        summary_df.to_csv('results/full_pipeline_summary.csv', index=False)
        print(f"\n   Summary saved to: results/full_pipeline_summary.csv")

        return summary_df


def main():
    """Run the full model building pipeline"""

    # Initialize pipeline
    pipeline = FullModelPipeline(random_state=42)

    # Run pipeline for a few targets (to avoid long runtime)
    # You can change this to process all targets or specific ones
    target_indices = [0, 7, 11]  # NR-AR, SR-ARE, SR-p53

    pipeline.run_full_pipeline(target_indices=target_indices, cv_folds=5)


if __name__ == "__main__":
    main()