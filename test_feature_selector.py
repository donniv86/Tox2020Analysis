#!/usr/bin/env python3
"""
Test script for FeatureSelector class with custom parameters
"""

import sys
import os
sys.path.append('src')

from feature_selector import FeatureSelector
from data_preparation import Tox21DataLoader
import numpy as np

def test_feature_selector_custom():
    """Test FeatureSelector with custom parameters for SR-ARE target"""

    print("=" * 60)
    print("TESTING FEATURESELECTOR WITH CUSTOM PARAMETERS")
    print("=" * 60)

    # Initialize data loader
    print("\n1. Loading data...")
    loader = Tox21DataLoader()
    loader.load_descriptors()
    loader.load_targets_from_sdf()
    loader.remove_low_variance_features(threshold=0.01)
    loader.handle_missing_values(strategy='drop')

    # Select SR-ARE target (index 7, better class balance)
    target_idx = 7  # SR-ARE
    target_name = loader.target_names[target_idx]
    print(f"\n2. Selected target: {target_name}")

    # Get target statistics
    target_stats = loader.get_target_statistics()
    sr_are_stats = target_stats[target_stats['target'] == target_name].iloc[0]
    print(f"   Total samples: {sr_are_stats['total_samples']}")
    print(f"   Active samples: {sr_are_stats['active_samples']}")
    print(f"   Active ratio: {sr_are_stats['active_ratio']:.3f}")

    # Prepare data
    y = loader.targets[:, target_idx]
    X = loader.descriptors
    feature_names = loader.feature_names

    print(f"\n3. Initial data shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")

    # Test with custom parameters
    print("\n4. Testing FeatureSelector with custom parameters...")

    # Custom parameters for better feature selection
    custom_params = {
        'correlation_threshold': 0.90,  # More aggressive correlation filtering
        'univariate_k': 500,           # More features from univariate selection
        'top_n_model': 150,            # More features from model-based selection
        'random_state': 42
    }

    print(f"   Correlation threshold: {custom_params['correlation_threshold']}")
    print(f"   Univariate k: {custom_params['univariate_k']}")
    print(f"   Top N model features: {custom_params['top_n_model']}")

    # Initialize and run feature selector
    selector = FeatureSelector(**custom_params)
    X_selected, selected_names = selector.fit_transform(X, y, feature_names)

    print(f"\n5. Feature selection results:")
    print(f"   Final shape: {X_selected.shape}")
    print(f"   Features reduced: {X.shape[1]} → {X_selected.shape[1]}")
    print(f"   Reduction ratio: {X_selected.shape[1]/X.shape[1]:.3f}")

    # Print selection history
    print(f"\n6. Selection history:")
    if 'correlation_filter' in selector.history_:
        corr_info = selector.history_['correlation_filter']
        print(f"   Correlation filter: Removed {len(corr_info['removed'])} features")

    if 'univariate_selection' in selector.history_:
        uni_info = selector.history_['univariate_selection']
        print(f"   Univariate selection: Selected {len(uni_info['selected'])} features")

    if 'model_based_selection' in selector.history_:
        model_info = selector.history_['model_based_selection']
        print(f"   Model-based selection: Selected {len(model_info['selected'])} features")

    # Plot feature importances
    print(f"\n7. Generating feature importance plot...")
    plot_path = f'results/{target_name}_feature_importances_custom.png'
    selector.plot_feature_importances(save_path=plot_path, top_n=20)

    # Save results
    print(f"\n8. Saving results...")
    features_path = f'results/{target_name}_selected_features_custom.npy'
    names_path = f'results/{target_name}_selected_feature_names_custom.npy'

    np.save(features_path, X_selected)
    np.save(names_path, selected_names)

    print(f"   Saved features to: {features_path}")
    print(f"   Saved feature names to: {names_path}")
    print(f"   Saved plot to: {plot_path}")

    # Show top 10 feature importances
    print(f"\n9. Top 10 feature importances:")
    for i, (name, importance) in enumerate(zip(selected_names[:10], selector.feature_importances_[:10])):
        print(f"   {i+1:2d}. {name}: {importance:.4f}")

    print("\n" + "=" * 60)
    print("✓ FEATURE SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return {
        'X_selected': X_selected,
        'selected_names': selected_names,
        'selector': selector,
        'target_name': target_name
    }

if __name__ == "__main__":
    test_feature_selector_custom()