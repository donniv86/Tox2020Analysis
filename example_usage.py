#!/usr/bin/env python3
"""
Example usage of the modular Tox21 modeling pipeline.

This script demonstrates different ways to use the clean, modular structure.
"""

import sys
from src.pipeline_manager import PipelineManager
from utils import Tox21Utils
from config import PIPELINE_CONFIG, DEFAULT_TARGET_INDICES

def example_1_basic_pipeline():
    """Example 1: Basic pipeline usage."""
    print("=" * 60)
    print("Example 1: Basic Pipeline Usage")
    print("=" * 60)

    # Initialize pipeline
    pipeline = PipelineManager(PIPELINE_CONFIG)

    # Run for specific targets
    results = pipeline.run_pipeline(target_indices=[0])  # Just NR-AR

    print(f"✅ Processed {len(results)} targets")
    return results

def example_2_custom_configuration():
    """Example 2: Custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Custom configuration
    custom_config = PIPELINE_CONFIG.copy()
    custom_config['cv_folds'] = 3  # Fewer CV folds for faster execution
    custom_config['models'] = ['RandomForest']  # Only RandomForest
    custom_config['feature_selection']['top_n_model'] = 100  # Fewer features

    print(f"Custom config: {custom_config['cv_folds']}-fold CV, {len(custom_config['models'])} models")

    # Initialize with custom config
    pipeline = PipelineManager(custom_config)

    # Run pipeline
    results = pipeline.run_pipeline(target_indices=[7])  # SR-ARE

    print(f"✅ Processed with custom configuration")
    return results

def example_3_utility_functions():
    """Example 3: Using utility functions."""
    print("\n" + "=" * 60)
    print("Example 3: Utility Functions")
    print("=" * 60)

    utils = Tox21Utils()

    # List available targets
    targets = utils.list_available_targets()
    print(f"Available targets: {targets}")

    if not targets:
        print("No trained models found. Run examples 1 or 2 first.")
        return

    # Get performance summary
    target_name = targets[0]
    summary = utils.get_performance_summary(target_name)
    print(f"\nPerformance summary for {target_name}:")
    for key, value in summary.items():
        if key != 'target_name':
            print(f"  {key}: {value}")

    # Compare targets (if multiple available)
    if len(targets) > 1:
        print(f"\nComparing {len(targets)} targets:")
        comparison = utils.compare_targets(targets)
        print(comparison[['target_name', 'best_model', 'roc_auc', 'f1_score']].to_string(index=False))

def example_4_model_loading():
    """Example 4: Loading and using trained models."""
    print("\n" + "=" * 60)
    print("Example 4: Model Loading")
    print("=" * 60)

    utils = Tox21Utils()
    targets = utils.list_available_targets()

    if not targets:
        print("No trained models found. Run examples 1 or 2 first.")
        return

    target_name = targets[0]

    # Load model and feature selector
    model_data = utils.load_model_and_features(target_name)
    print(f"✅ Loaded model for {target_name}")
    print(f"   Model type: {type(model_data['model']).__name__}")
    print(f"   Selected features: {len(model_data['selected_features'])}")

    # Example: Create dummy data for prediction (in real usage, this would be actual molecular descriptors)
    import numpy as np
    n_samples = 5
    n_features = len(model_data['selected_features'])

    # Create dummy descriptors (this is just for demonstration)
    dummy_descriptors = np.random.randn(n_samples, n_features)
    dummy_feature_names = model_data['selected_features']

    try:
        # Make predictions
        predictions = utils.predict_toxicity(target_name, dummy_descriptors, dummy_feature_names)
        print(f"✅ Made predictions for {n_samples} compounds")
        print(f"   Predictions: {predictions['predictions']}")
        print(f"   Probabilities: {predictions['probabilities'][:3]}...")  # Show first 3
    except Exception as e:
        print(f"⚠️  Prediction failed (expected with dummy data): {str(e)}")

def example_5_configuration_exploration():
    """Example 5: Exploring configuration options."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration Exploration")
    print("=" * 60)

    print("Available configuration options:")
    print(f"  Models: {PIPELINE_CONFIG['models']}")
    print(f"  CV folds: {PIPELINE_CONFIG['cv_folds']}")
    print(f"  Feature selection: {PIPELINE_CONFIG['feature_selection']}")
    print(f"  Min samples: {PIPELINE_CONFIG['min_samples']}")
    print(f"  Min positive samples: {PIPELINE_CONFIG['min_positive_samples']}")

    print(f"\nDefault target indices: {DEFAULT_TARGET_INDICES}")
    print(f"All available targets: {len(PIPELINE_CONFIG.get('ALL_TARGETS', []))} targets")

def main():
    """Run all examples."""
    print("🚀 Tox21 Modeling Pipeline - Example Usage")
    print("This script demonstrates the modular pipeline structure.\n")

    try:
        # Run examples
        example_1_basic_pipeline()
        example_2_custom_configuration()
        example_3_utility_functions()
        example_4_model_loading()
        example_5_configuration_exploration()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("📚 Check MODULAR_STRUCTURE.md for detailed documentation")
        print("🔧 Modify config.py to experiment with different settings")

    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Examples failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())