#!/usr/bin/env python3
"""
Cross-validation test for baseline models
"""

import sys
import os
sys.path.append('src')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

from data_preparation import Tox21DataLoader
from feature_selector import FeatureSelector

def test_baseline_cv():
    """Test baseline models with proper cross-validation"""

    print("=" * 60)
    print("CROSS-VALIDATION BASELINE MODEL TEST")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    loader = Tox21DataLoader()
    loader.load_descriptors()
    loader.load_targets_from_sdf()
    loader.remove_low_variance_features(threshold=0.01)
    loader.handle_missing_values(strategy='drop')

    # Select target
    target_idx = 7  # SR-ARE
    target_name = loader.target_names[target_idx]
    print(f"\n2. Target: {target_name}")

    # Feature selection
    print("\n3. Feature selection...")
    y = loader.targets[:, target_idx]
    X = loader.descriptors
    feature_names = loader.feature_names

    selector = FeatureSelector(correlation_threshold=0.90, univariate_k=500, top_n_model=150)
    X_selected, selected_names = selector.fit_transform(X, y, feature_names)
    print(f"   Selected features: {X_selected.shape[1]}")

    # Cross-validation setup
    print("\n4. Cross-validation setup...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=42, class_weight='balanced'
        )
    }

    # Train and evaluate with CV
    print("\n5. Training models with cross-validation...")
    results = {name: {'roc_auc': [], 'pr_auc': [], 'f1': [], 'precision': [], 'recall': []} for name in models.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y)):
        print(f"\n   Fold {fold_idx + 1}:")
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"     Train: {X_train.shape[0]} samples ({np.sum(y_train)} positive)")
        print(f"     Test:  {X_test.shape[0]} samples ({np.sum(y_test)} positive)")

        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            # Store results
            results[name]['roc_auc'].append(roc_auc)
            results[name]['pr_auc'].append(pr_auc)
            results[name]['f1'].append(f1)
            results[name]['precision'].append(precision)
            results[name]['recall'].append(recall)

            print(f"     {name}: ROC-AUC = {roc_auc:.3f}, PR-AUC = {pr_auc:.3f}, F1 = {f1:.3f}")

    # Calculate average results
    print(f"\n6. Cross-validation Results Summary:")
    print(f"{'Model':<15} {'ROC-AUC':<12} {'PR-AUC':<12} {'F1-Score':<12}")
    print("-" * 55)

    summary_data = []
    for name, metrics in results.items():
        roc_auc_mean = np.mean(metrics['roc_auc'])
        roc_auc_std = np.std(metrics['roc_auc'])
        pr_auc_mean = np.mean(metrics['pr_auc'])
        pr_auc_std = np.std(metrics['pr_auc'])
        f1_mean = np.mean(metrics['f1'])
        f1_std = np.std(metrics['f1'])

        print(f"{name:<15} {roc_auc_mean:.3f}±{roc_auc_std:.3f} {pr_auc_mean:.3f}±{pr_auc_std:.3f} {f1_mean:.3f}±{f1_std:.3f}")

        summary_data.append({
            'Model': name,
            'ROC-AUC_Mean': roc_auc_mean,
            'ROC-AUC_Std': roc_auc_std,
            'PR-AUC_Mean': pr_auc_mean,
            'PR-AUC_Std': pr_auc_std,
            'F1_Mean': f1_mean,
            'F1_Std': f1_std
        })

    # Save results
    print(f"\n7. Saving results...")
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/SR-ARE_baseline_cv_results.csv', index=False)
    print(f"   Saved to: results/SR-ARE_baseline_cv_results.csv")

    # Find best model
    best_model = summary_df.loc[summary_df['ROC-AUC_Mean'].idxmax(), 'Model']
    print(f"\n8. Best model by ROC-AUC: {best_model}")

    print("\n" + "=" * 60)
    print("✓ CROSS-VALIDATION BASELINE TEST COMPLETED!")
    print("=" * 60)

    return summary_df

if __name__ == "__main__":
    test_baseline_cv()