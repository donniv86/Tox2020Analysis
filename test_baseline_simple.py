#!/usr/bin/env python3
"""
Simple test script for baseline models
"""

import sys
import os
sys.path.append('src')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from data_preparation import Tox21DataLoader
from feature_selector import FeatureSelector

def test_baseline_simple():
    """Simple test of baseline models with train/test split"""

    print("=" * 60)
    print("SIMPLE BASELINE MODEL TEST")
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

    # Simple train/test split
    print("\n4. Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Define models
    print("\n5. Training models...")
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

    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\n   {name}:")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        results[name] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

        print(f"     ROC-AUC: {roc_auc:.3f}")
        print(f"     PR-AUC:  {pr_auc:.3f}")

    # Print summary
    print(f"\n6. Results Summary:")
    print(f"{'Model':<15} {'ROC-AUC':<10} {'PR-AUC':<10}")
    print("-" * 35)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['roc_auc']:<10.3f} {metrics['pr_auc']:<10.3f}")

    print("\n" + "=" * 60)
    print("✓ SIMPLE BASELINE TEST COMPLETED!")
    print("=" * 60)

    return results

if __name__ == "__main__":
    test_baseline_simple()