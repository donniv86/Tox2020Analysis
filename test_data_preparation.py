#!/usr/bin/env python3
"""
Test script for data preparation
"""

import sys
import os
sys.path.append('src')

from data_preparation import Tox21DataLoader

def test_data_preparation():
    """Test the data preparation pipeline"""

    print("=" * 60)
    print("TOX21 DATA PREPARATION TEST")
    print("=" * 60)

    # Initialize data loader
    print("\n1. Initializing data loader...")
    loader = Tox21DataLoader()

    # Load descriptors
    print("\n2. Loading descriptors...")
    try:
        descriptors = loader.load_descriptors()
        print(f"✓ Successfully loaded descriptors: {descriptors.shape}")
    except Exception as e:
        print(f"✗ Error loading descriptors: {e}")
        return

    # Load targets
    print("\n3. Loading target labels...")
    try:
        targets = loader.load_targets_from_sdf()
        print(f"✓ Successfully loaded targets: {targets.shape}")
    except Exception as e:
        print(f"✗ Error loading targets: {e}")
        return

    # Get target statistics
    print("\n4. Analyzing target statistics...")
    try:
        target_stats = loader.get_target_statistics()
        print("\nTarget Statistics:")
        print(target_stats.to_string(index=False))
    except Exception as e:
        print(f"✗ Error getting target statistics: {e}")
        return

    # Remove low variance features
    print("\n5. Removing low variance features...")
    try:
        loader.remove_low_variance_features(threshold=0.01)
        print(f"✓ Features after variance thresholding: {loader.descriptors.shape[1]}")
    except Exception as e:
        print(f"✗ Error in variance thresholding: {e}")
        return

    # Handle missing values
    print("\n6. Handling missing values...")
    try:
        loader.handle_missing_values(strategy='drop')
        print(f"✓ Samples after handling missing values: {loader.descriptors.shape[0]}")
    except Exception as e:
        print(f"✗ Error handling missing values: {e}")
        return

    # Get data summary
    print("\n7. Data summary...")
    try:
        summary = loader.get_data_summary()
        print("\nData Summary:")
        for key, value in summary.items():
            if key != 'target_statistics':
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Error getting data summary: {e}")
        return

    # Test data preparation for first target
    print("\n8. Testing data preparation for first target...")
    try:
        data_dict = loader.prepare_data_for_target(
            target_idx=0,
            handle_imbalance=True,
            scale_features=True
        )

        print(f"\n✓ Successfully prepared data for {data_dict['target_name']}")
        print(f"  Training samples: {data_dict['X_train'].shape[0]}")
        print(f"  Validation samples: {data_dict['X_val'].shape[0]}")
        print(f"  Test samples: {data_dict['X_test'].shape[0]}")
        print(f"  Features: {data_dict['X_train'].shape[1]}")
        print(f"  Class weights: {data_dict['class_weights']}")

    except Exception as e:
        print(f"✗ Error preparing data for target: {e}")
        return

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    # Save prepared data for later use
    print("\n9. Saving prepared data...")
    try:
        import numpy as np
        np.save('results/prepared_data_target_0.npy', data_dict)
        print("✓ Saved prepared data to results/prepared_data_target_0.npy")
    except Exception as e:
        print(f"✗ Error saving data: {e}")

if __name__ == "__main__":
    test_data_preparation()