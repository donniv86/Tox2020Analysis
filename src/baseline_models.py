"""
Baseline Models for Tox21 Dataset

This module provides:
1. Multiple baseline models (RF, XGBoost, SVM, Logistic Regression)
2. Cross-validation evaluation
3. Performance metrics for imbalanced data
4. Model comparison and visualization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """
    Baseline models for Tox21 toxicity prediction
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def get_models(self, use_class_weights=True):
        """
        Get baseline models with default parameters

        Args:
            use_class_weights: Whether to use class weights for imbalanced data
        """
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced' if use_class_weights else None
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced' if use_class_weights else None
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced' if use_class_weights else None
            )
        }
        return models

    def train_and_evaluate_cv(self, cv_splits, model_name, model, target_name):
        """
        Train and evaluate a model using cross-validation splits

        Args:
            cv_splits: List of CV splits from Tox21CrossValidation
            model_name: Name of the model
            model: Model instance
            target_name: Name of the target

        Returns:
            Dictionary with results for each fold
        """
        print(f"Training {model_name} for {target_name}...")

        fold_results = []

        for fold_idx, split in enumerate(cv_splits):
            X_train = split['X_train']
            X_test = split['X_test']
            y_train = split['y_train']
            y_test = split['y_test']

            # Note: SMOTE is already applied in the CV preparation
            # So we don't need to apply it again here

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['fold'] = fold_idx

            fold_results.append({
                'fold': fold_idx,
                'model': model,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })

            print(f"  Fold {fold_idx + 1}: ROC-AUC = {metrics['roc_auc']:.3f}, PR-AUC = {metrics['pr_auc']:.3f}")

        return fold_results

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive metrics for imbalanced classification

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true == 1),
            'n_negative': np.sum(y_true == 0)
        }

        # Calculate balanced accuracy
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['balanced_accuracy'] = (tp / (tp + fn) + tn / (tn + fp)) / 2

        return metrics

    def evaluate_all_models(self, cv_data, target_name):
        """
        Evaluate all baseline models on a target

        Args:
            cv_data: Cross-validation data from Tox21CrossValidation
            target_name: Name of the target

        Returns:
            Dictionary with results for all models
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING BASELINE MODELS FOR {target_name}")
        print(f"{'='*60}")

        cv_splits = cv_data['cv_splits']
        models = self.get_models(use_class_weights=True)

        all_results = {}

        for model_name, model in models.items():
            print(f"\n{model_name}:")
            fold_results = self.train_and_evaluate_cv(cv_splits, model_name, model, target_name)
            all_results[model_name] = fold_results

            # Calculate average metrics across folds
            avg_metrics = self.calculate_average_metrics(fold_results)
            print(f"  Average ROC-AUC: {avg_metrics['roc_auc']:.3f} ± {avg_metrics['roc_auc_std']:.3f}")
            print(f"  Average PR-AUC:  {avg_metrics['pr_auc']:.3f} ± {avg_metrics['pr_auc_std']:.3f}")

        self.results[target_name] = all_results
        return all_results

    def calculate_average_metrics(self, fold_results):
        """
        Calculate average metrics across CV folds

        Args:
            fold_results: List of results from each fold

        Returns:
            Dictionary with average metrics and standard deviations
        """
        metrics_list = [result['metrics'] for result in fold_results]

        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'balanced_accuracy']:
            values = [m[metric] for m in metrics_list]
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)

        return avg_metrics

    def compare_models(self, target_name):
        """
        Compare all models for a target

        Args:
            target_name: Name of the target

        Returns:
            DataFrame with comparison results
        """
        if target_name not in self.results:
            print(f"No results found for {target_name}")
            return None

        comparison_data = []

        for model_name, fold_results in self.results[target_name].items():
            avg_metrics = self.calculate_average_metrics(fold_results)
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{avg_metrics['roc_auc']:.3f} ± {avg_metrics['roc_auc_std']:.3f}",
                'PR-AUC': f"{avg_metrics['pr_auc']:.3f} ± {avg_metrics['pr_auc_std']:.3f}",
                'F1-Score': f"{avg_metrics['f1']:.3f} ± {avg_metrics['f1_std']:.3f}",
                'Balanced_Accuracy': f"{avg_metrics['balanced_accuracy']:.3f} ± {avg_metrics['balanced_accuracy_std']:.3f}",
                'ROC-AUC_Mean': avg_metrics['roc_auc'],
                'PR-AUC_Mean': avg_metrics['pr_auc']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC_Mean', ascending=False)

        return comparison_df

    def plot_model_comparison(self, target_name, save_path=None):
        """
        Plot model comparison for a target

        Args:
            target_name: Name of the target
            save_path: Path to save the plot
        """
        comparison_df = self.compare_models(target_name)
        if comparison_df is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ROC-AUC comparison
        models = comparison_df['Model']
        roc_auc_means = comparison_df['ROC-AUC_Mean']
        roc_auc_stds = [float(x.split('±')[1].strip()) for x in comparison_df['ROC-AUC']]

        ax1.bar(models, roc_auc_means, yerr=roc_auc_stds, capsize=5, alpha=0.7)
        ax1.set_title(f'ROC-AUC Comparison - {target_name}')
        ax1.set_ylabel('ROC-AUC')
        ax1.tick_params(axis='x', rotation=45)

        # PR-AUC comparison
        pr_auc_means = comparison_df['PR-AUC_Mean']
        pr_auc_stds = [float(x.split('±')[1].strip()) for x in comparison_df['PR-AUC']]

        ax2.bar(models, pr_auc_means, yerr=pr_auc_stds, capsize=5, alpha=0.7)
        ax2.set_title(f'PR-AUC Comparison - {target_name}')
        ax2.set_ylabel('PR-AUC')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return comparison_df

    def save_results(self, target_name, save_dir='results'):
        """
        Save model results to files

        Args:
            target_name: Name of the target
            save_dir: Directory to save results
        """
        import os
        import pickle

        if target_name not in self.results:
            print(f"No results found for {target_name}")
            return

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Save detailed results
        results_path = os.path.join(save_dir, f'{target_name}_baseline_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results[target_name], f)

        # Save comparison table
        comparison_df = self.compare_models(target_name)
        if comparison_df is not None:
            csv_path = os.path.join(save_dir, f'{target_name}_model_comparison.csv')
            comparison_df.to_csv(csv_path, index=False)

        # Save plot
        plot_path = os.path.join(save_dir, f'{target_name}_model_comparison.png')
        self.plot_model_comparison(target_name, save_path=plot_path)

        print(f"Results saved to {save_dir}/")
        print(f"  - Detailed results: {results_path}")
        print(f"  - Comparison table: {csv_path}")
        print(f"  - Comparison plot: {plot_path}")


def main():
    """Example usage of BaselineModels"""

    # Import required modules
    from data_preparation import Tox21DataLoader
    from cross_validation_preparation import Tox21CrossValidation
    from feature_selector import FeatureSelector

    # Load and prepare data
    print("Loading data...")
    loader = Tox21DataLoader()
    loader.load_descriptors()
    loader.load_targets_from_sdf()
    loader.remove_low_variance_features(threshold=0.01)
    loader.handle_missing_values(strategy='drop')

    # Select target (SR-ARE for better class balance)
    target_idx = 7  # SR-ARE
    target_name = loader.target_names[target_idx]

    # Feature selection
    print(f"Performing feature selection for {target_name}...")
    y = loader.targets[:, target_idx]
    X = loader.descriptors
    feature_names = loader.feature_names

    selector = FeatureSelector(correlation_threshold=0.90, univariate_k=500, top_n_model=150)
    X_selected, selected_names = selector.fit_transform(X, y, feature_names)

    # Prepare cross-validation
    print("Preparing cross-validation...")
    cv_prep = Tox21CrossValidation(loader)
    cv_data = cv_prep.prepare_stratified_kfold(target_idx=target_idx, n_splits=5)

    # Update CV splits with selected features
    for split in cv_data['cv_splits']:
        # Get original indices
        train_indices = split['train_indices']
        test_indices = split['test_indices']

        # Update with selected features
        split['X_train'] = X_selected[train_indices]
        split['X_test'] = X_selected[test_indices]

    # Train and evaluate baseline models
    print("Training baseline models...")
    baseline_models = BaselineModels()
    results = baseline_models.evaluate_all_models(cv_data, target_name)

    # Compare and save results
    comparison_df = baseline_models.compare_models(target_name)
    print(f"\nModel Comparison for {target_name}:")
    print(comparison_df[['Model', 'ROC-AUC', 'PR-AUC', 'F1-Score']].to_string(index=False))

    # Save results
    baseline_models.save_results(target_name)


if __name__ == "__main__":
    main()