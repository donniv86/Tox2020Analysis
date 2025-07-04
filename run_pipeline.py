#!/usr/bin/env python3
"""
Simple script to run the Tox21 modeling pipeline using the new modular structure.

This script demonstrates how to use the clean, modular pipeline classes.
"""

import sys
from src.pipeline_manager import PipelineManager
from config import PIPELINE_CONFIG, DEFAULT_TARGET_INDICES

def main():
    """Run the Tox21 modeling pipeline."""

    # Initialize pipeline manager with configuration
    pipeline = PipelineManager(PIPELINE_CONFIG)

    try:
        print("🚀 Starting Tox21 modeling pipeline...")
        print(f"📋 Configuration: {len(PIPELINE_CONFIG['models'])} models, {PIPELINE_CONFIG['cv_folds']}-fold CV")
        print(f"🎯 Targets: {len(DEFAULT_TARGET_INDICES)} targets")

        results = pipeline.run_pipeline(target_indices=DEFAULT_TARGET_INDICES)

        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(results)} targets")
        print(f"📁 Results saved in '{PIPELINE_CONFIG['paths']['results_dir']}/' directory")
        print(f"📝 Logs saved in '{PIPELINE_CONFIG['paths']['logs_dir']}/' directory")

        # Print summary
        print("\n📈 Performance Summary:")
        for target_name, target_results in results.items():
            best_model = target_results['best_model']
            best_metrics = target_results['models'][best_model]['avg_metrics']
            roc_auc = best_metrics['roc_auc_mean']
            print(f"  {target_name}: {best_model} (ROC-AUC: {roc_auc:.3f})")

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user. Progress saved.")
        print("   Resume later by running the script again.")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("   Check logs for detailed error information.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())