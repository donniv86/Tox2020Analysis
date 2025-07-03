#!/usr/bin/env python3
"""
Main training script for Tox21 toxicity prediction models.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append('src')

from data_processing import Tox21DataLoader
from feature_engineering import MolecularFeatureGenerator
from models import ToxicityPredictor, EnsemblePredictor
from evaluation import ModelEvaluator


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("TOX21 TOXICITY PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    loader = Tox21DataLoader("data/tox21_10k_data_all.sdf")
    data = loader.load_data()

    # Print data summary
    summary = loader.get_data_summary()
    print(f"Loaded {summary['total_compounds']} compounds")
    for target, stats in summary['targets'].items():
        print(f"  {target}: {stats['positive_samples']}/{stats['total_samples']} positive ({stats['positive_ratio']:.2%})")

    # Step 2: Generate molecular features
    print("\n2. Generating molecular features...")
    feature_gen = MolecularFeatureGenerator()

    # Generate features
    X = feature_gen.generate_features(
        data['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    print(f"Generated {X.shape[1]} features for {X.shape[0]} compounds")

    # Step 3: Split data
    print("\n3. Splitting data into train/test sets...")
    splits = loader.split_data(test_size=0.2, random_state=42)

    X_train = feature_gen.generate_features(
        splits['train']['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )
    X_test = feature_gen.generate_features(
        splits['test']['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    y_train = splits['train']['targets']
    y_test = splits['test']['targets']

    print(f"Train set: {X_train.shape[0]} compounds")
    print(f"Test set: {X_test.shape[0]} compounds")

    # Step 4: Train models
    print("\n4. Training models...")

    # Train individual models
    models = {}
    for model_type in ['random_forest', 'logistic', 'svm']:
        print(f"\nTraining {model_type}...")
        predictor = ToxicityPredictor(model_type)
        predictor.train(X_train, y_train)
        models[model_type] = predictor

    # Train ensemble
    print("\nTraining ensemble model...")
    ensemble = EnsemblePredictor(['random_forest', 'logistic', 'svm'])
    ensemble.train(X_train, y_train)
    models['ensemble'] = ensemble

    # Step 5: Evaluate models
    print("\n5. Evaluating models...")
    evaluator = ModelEvaluator()

    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        # Make predictions
        if model_name == 'ensemble':
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = pd.DataFrame()
            for target in y_pred_proba.columns:
                y_pred[target] = (y_pred_proba[target] > 0.5).astype(int)

        # Evaluate
        results_df = evaluator.evaluate_all_targets(y_test, y_pred, y_pred_proba)
        results[model_name] = results_df

        # Print summary
        print(f"  Average ROC-AUC: {results_df['roc_auc'].mean():.3f}")
        print(f"  Average PR-AUC: {results_df['pr_auc'].mean():.3f}")
        print(f"  Average F1 Score: {results_df['f1_score'].mean():.3f}")

    # Step 6: Save results
    print("\n6. Saving results...")

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Save models
    for model_name, model in models.items():
        if model_name != 'ensemble':
            model.save_models(f'models/{model_name}_models.pkl')

    # Save results
    for model_name, result_df in results.items():
        result_df.to_csv(f'results/{model_name}_results.csv', index=False)

    # Generate and save comprehensive report
    print("\n7. Generating comprehensive report...")

    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['roc_auc'].mean())
    best_results = results[best_model]

    report = evaluator.generate_report(best_results)

    with open('results/evaluation_report.txt', 'w') as f:
        f.write(report)

    print(f"\nBest model: {best_model}")
    print(f"Average ROC-AUC: {best_results['roc_auc'].mean():.3f}")

    # Step 8: Generate plots
    print("\n8. Generating plots...")

    # Get best model predictions
    best_model_instance = models[best_model]
    if best_model == 'ensemble':
        y_pred_proba_best = best_model_instance.predict(X_test)
    else:
        y_pred_proba_best = best_model_instance.predict(X_test)

    # Plot ROC curves
    evaluator.plot_roc_curves(y_test, y_pred_proba_best, 'results/roc_curves.png')

    # Plot PR curves
    evaluator.plot_pr_curves(y_test, y_pred_proba_best, 'results/pr_curves.png')

    # Plot performance summary
    evaluator.plot_performance_summary(best_results, 'results/performance_summary.png')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nResults saved in 'results/' directory")
    print("Models saved in 'models/' directory")
    print(f"Best model: {best_model}")
    print(f"Best average ROC-AUC: {best_results['roc_auc'].mean():.3f}")


if __name__ == "__main__":
    main()