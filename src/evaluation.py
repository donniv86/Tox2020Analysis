"""
Evaluation module for model performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation class."""

    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {}

        # Basic metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['precision'] = report['1']['precision']
        metrics['recall'] = report['1']['recall']
        metrics['f1_score'] = report['1']['f1-score']
        metrics['accuracy'] = report['accuracy']

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]

        # Additional metrics
        metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2

        return metrics

    def evaluate_all_targets(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                           y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        """Evaluate performance across all targets."""
        results = []

        for target in y_true.columns:
            if target in y_pred.columns and target in y_pred_proba.columns:
                # Get data for this target
                mask = y_true[target].notna()
                if mask.sum() > 0:
                    y_t = y_true[target][mask].values
                    y_p = y_pred[target][mask].values
                    y_pp = y_pred_proba[target][mask].values

                    # Calculate metrics
                    metrics = self.calculate_metrics(y_t, y_p, y_pp)
                    metrics['target'] = target
                    results.append(metrics)

        return pd.DataFrame(results)

    def plot_roc_curves(self, y_true: pd.DataFrame, y_pred_proba: pd.DataFrame,
                       save_path: str = None):
        """Plot ROC curves for all targets."""
        plt.figure(figsize=(12, 8))

        for target in y_true.columns:
            if target in y_pred_proba.columns:
                mask = y_true[target].notna()
                if mask.sum() > 0:
                    y_t = y_true[target][mask].values
                    y_pp = y_pred_proba[target][mask].values

                    fpr, tpr, _ = roc_curve(y_t, y_pp)
                    auc = roc_auc_score(y_t, y_pp)

                    plt.plot(fpr, tpr, label=f'{target} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Targets')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pr_curves(self, y_true: pd.DataFrame, y_pred_proba: pd.DataFrame,
                      save_path: str = None):
        """Plot Precision-Recall curves for all targets."""
        plt.figure(figsize=(12, 8))

        for target in y_true.columns:
            if target in y_pred_proba.columns:
                mask = y_true[target].notna()
                if mask.sum() > 0:
                    y_t = y_true[target][mask].values
                    y_pp = y_pred_proba[target][mask].values

                    precision, recall, _ = precision_recall_curve(y_t, y_pp)
                    auc = average_precision_score(y_t, y_pp)

                    plt.plot(recall, precision, label=f'{target} (AUC = {auc:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Targets')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_summary(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot performance summary across targets."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC-AUC
        axes[0, 0].bar(results_df['target'], results_df['roc_auc'])
        axes[0, 0].set_title('ROC-AUC by Target')
        axes[0, 0].set_ylabel('ROC-AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # PR-AUC
        axes[0, 1].bar(results_df['target'], results_df['pr_auc'])
        axes[0, 1].set_title('PR-AUC by Target')
        axes[0, 1].set_ylabel('PR-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # F1 Score
        axes[1, 0].bar(results_df['target'], results_df['f1_score'])
        axes[1, 0].set_title('F1 Score by Target')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Balanced Accuracy
        axes[1, 1].bar(results_df['target'], results_df['balanced_accuracy'])
        axes[1, 1].set_title('Balanced Accuracy by Target')
        axes[1, 1].set_ylabel('Balanced Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("TOX21 MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall statistics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"Average ROC-AUC: {results_df['roc_auc'].mean():.3f} ± {results_df['roc_auc'].std():.3f}")
        report.append(f"Average PR-AUC: {results_df['pr_auc'].mean():.3f} ± {results_df['pr_auc'].std():.3f}")
        report.append(f"Average F1 Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
        report.append("")

        # Best performing targets
        report.append("BEST PERFORMING TARGETS (by ROC-AUC):")
        best_targets = results_df.nlargest(3, 'roc_auc')[['target', 'roc_auc', 'pr_auc']]
        for _, row in best_targets.iterrows():
            report.append(f"  {row['target']}: ROC-AUC={row['roc_auc']:.3f}, PR-AUC={row['pr_auc']:.3f}")
        report.append("")

        # Worst performing targets
        report.append("WORST PERFORMING TARGETS (by ROC-AUC):")
        worst_targets = results_df.nsmallest(3, 'roc_auc')[['target', 'roc_auc', 'pr_auc']]
        for _, row in worst_targets.iterrows():
            report.append(f"  {row['target']}: ROC-AUC={row['roc_auc']:.3f}, PR-AUC={row['pr_auc']:.3f}")
        report.append("")

        # Detailed results table
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)
        report.append(f"{'Target':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<8} {'Bal_Acc':<8}")
        report.append("-" * 80)

        for _, row in results_df.iterrows():
            report.append(f"{row['target']:<20} {row['roc_auc']:<10.3f} {row['pr_auc']:<10.3f} "
                         f"{row['f1_score']:<8.3f} {row['balanced_accuracy']:<8.3f}")

        return "\n".join(report)