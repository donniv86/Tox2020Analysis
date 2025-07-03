#!/usr/bin/env python3
"""
Example usage of the Toxicity Prediction System
"""

import pandas as pd
import os
import sys

# Add src to path
sys.path.append('src')

from data_processing import Tox21DataLoader
from feature_engineering import MolecularFeatureGenerator
from models import ToxicityPredictor


def create_example_data():
    """Create example SMILES data for testing."""
    example_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Gefitinib
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # Aspirin
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Gefitinib
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'compound_id': [f'compound_{i+1}' for i in range(len(example_smiles))],
        'smiles': example_smiles
    })

    # Save to CSV
    os.makedirs('example_data', exist_ok=True)
    df.to_csv('example_data/example_compounds.csv', index=False)
    print("Created example data: example_data/example_compounds.csv")

    return df


def train_models():
    """Train models on Tox21 data."""
    print("Training models on Tox21 data...")

    # Load Tox21 data
    loader = Tox21DataLoader("data/tox21_10k_data_all.sdf")
    data = loader.load_data()

    # Generate features
    feature_gen = MolecularFeatureGenerator()
    X = feature_gen.generate_features(
        data['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    # Split data
    splits = loader.split_data(test_size=0.2, random_state=42)

    X_train = feature_gen.generate_features(
        splits['train']['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    y_train = splits['train']['targets']

    # Train model
    predictor = ToxicityPredictor('random_forest')
    predictor.train(X_train, y_train)

    # Save models
    os.makedirs('models', exist_ok=True)
    predictor.save_models('models/random_forest_models.pkl')
    print("Models trained and saved to: models/random_forest_models.pkl")

    return predictor


def predict_toxicity_example():
    """Example of toxicity prediction."""
    print("\n" + "="*60)
    print("TOXICITY PREDICTION EXAMPLE")
    print("="*60)

    # Create example data
    example_df = create_example_data()

    # Train models (or load if available)
    if os.path.exists('models/random_forest_models.pkl'):
        print("Loading pre-trained models...")
        predictor = ToxicityPredictor('random_forest')
        predictor.load_models('models/random_forest_models.pkl')
    else:
        print("Training new models...")
        predictor = train_models()

    # Generate features for example compounds
    feature_gen = MolecularFeatureGenerator()
    X_example = feature_gen.generate_features(
        example_df['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    # Make predictions
    predictions = predictor.predict(X_example)

    # Create results
    results = example_df.copy()
    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    for target in target_columns:
        if target in predictions.columns:
            results[f'{target}_probability'] = predictions[target]
            results[f'{target}_prediction'] = (predictions[target] > 0.5).astype(int)
            results[f'{target}_toxicity'] = classify_toxicity(predictions[target])

    # Display results
    print("\nToxicity Predictions:")
    print("-" * 80)

    for idx, row in results.iterrows():
        print(f"\nCompound: {row['compound_id']}")
        print(f"SMILES: {row['smiles']}")
        print("Toxicity Predictions:")

        toxic_count = 0
        moderate_count = 0
        non_toxic_count = 0

        for target in target_columns:
            tox_col = f'{target}_toxicity'
            prob_col = f'{target}_probability'

            if tox_col in row and prob_col in row:
                toxicity = row[tox_col]
                probability = row[prob_col]

                if toxicity == 'Toxic':
                    toxic_count += 1
                    color = '🔴'
                elif toxicity == 'Moderate':
                    moderate_count += 1
                    color = '🟡'
                else:
                    non_toxic_count += 1
                    color = '🟢'

                print(f"  {color} {target}: {toxicity} ({probability:.3f})")

        # Overall summary
        print(f"\nOverall: {toxic_count} Toxic, {moderate_count} Moderate, {non_toxic_count} Non-toxic")

        if toxic_count > 0:
            print("⚠️  WARNING: This compound shows toxic properties!")
        elif moderate_count > 0:
            print("⚠️  CAUTION: This compound shows moderate toxicity.")
        else:
            print("✅ SAFE: This compound appears non-toxic.")

    # Save detailed results
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/toxicity_predictions_example.csv', index=False)
    print(f"\nDetailed results saved to: results/toxicity_predictions_example.csv")

    return results


def classify_toxicity(probabilities):
    """Classify toxicity based on probability thresholds."""
    classifications = []
    for prob in probabilities:
        if prob < 0.3:
            classifications.append('Non-toxic')
        elif prob < 0.7:
            classifications.append('Moderate')
        else:
            classifications.append('Toxic')
    return classifications


def create_simple_report(results_df):
    """Create a simple text report."""
    print("\n" + "="*60)
    print("TOXICITY PREDICTION REPORT")
    print("="*60)

    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    # Summary statistics
    total_compounds = len(results_df)
    print(f"\nTotal Compounds Analyzed: {total_compounds}")

    # Target-wise summary
    print("\nTarget-wise Toxicity Summary:")
    print("-" * 50)

    for target in target_columns:
        tox_col = f'{target}_toxicity'
        if tox_col in results_df.columns:
            toxicity_counts = results_df[tox_col].value_counts()
            print(f"\n{target}:")
            for toxicity, count in toxicity_counts.items():
                print(f"  {toxicity}: {count} compounds")

    # Overall compound summary
    print("\nCompound-wise Summary:")
    print("-" * 50)

    for idx, row in results_df.iterrows():
        toxic_count = 0
        moderate_count = 0
        non_toxic_count = 0

        for target in target_columns:
            tox_col = f'{target}_toxicity'
            if tox_col in row and pd.notna(row[tox_col]):
                if row[tox_col] == 'Toxic':
                    toxic_count += 1
                elif row[tox_col] == 'Moderate':
                    moderate_count += 1
                else:
                    non_toxic_count += 1

        print(f"\n{row['compound_id']}:")
        print(f"  Toxic targets: {toxic_count}")
        print(f"  Moderate targets: {moderate_count}")
        print(f"  Non-toxic targets: {non_toxic_count}")

        if toxic_count > 0:
            print("  ⚠️  OVERALL: TOXIC")
        elif moderate_count > 0:
            print("  ⚠️  OVERALL: MODERATE TOXICITY")
        else:
            print("  ✅ OVERALL: SAFE")


if __name__ == "__main__":
    # Run the example
    results = predict_toxicity_example()
    create_simple_report(results)

    print("\n" + "="*60)
    print("EXAMPLE COMPLETED!")
    print("="*60)
    print("\nTo use with your own data:")
    print("1. Create a CSV file with 'compound_id' and 'smiles' columns")
    print("2. Run: python toxicity_predictor.py your_file.csv")
    print("3. Check the results/ directory for detailed reports")