#!/usr/bin/env python3
"""
Simple Command-Line Toxicity Predictor
Usage: python simple_predictor.py input.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing import Tox21DataLoader
from feature_engineering import MolecularFeatureGenerator
from models import ToxicityPredictor


def load_or_train_models():
    """Load pre-trained models or train new ones."""
    if os.path.exists('models/random_forest_models.pkl'):
        print("Loading pre-trained models...")
        predictor = ToxicityPredictor('random_forest')
        predictor.load_models('models/random_forest_models.pkl')
        return predictor
    else:
        print("Training new models...")
        return train_models()


def train_models():
    """Train models on Tox21 data."""
    print("Loading Tox21 data...")
    loader = Tox21DataLoader("data/tox21_10k_data_all.sdf")
    data = loader.load_data()

    print("Generating features...")
    feature_gen = MolecularFeatureGenerator()
    X = feature_gen.generate_features(
        data['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    print("Splitting data...")
    splits = loader.split_data(test_size=0.2, random_state=42)

    X_train = feature_gen.generate_features(
        splits['train']['compounds']['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    y_train = splits['train']['targets']

    print("Training Random Forest model...")
    predictor = ToxicityPredictor('random_forest')
    predictor.train(X_train, y_train)

    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    predictor.save_models('models/random_forest_models.pkl')

    return predictor


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


def load_input_file(file_path):
    """Load input file (CSV, SDF, or MOL)."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)

        # Check for SMILES column
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break

        if smiles_col is None:
            raise ValueError("No SMILES column found in CSV file")

        # Create molecules dataframe
        molecules = pd.DataFrame()
        molecules['compound_id'] = df.get('compound_id', range(len(df)))
        molecules['smiles'] = df[smiles_col]

        return molecules

    elif file_path.suffix.lower() == '.sdf':
        from rdkit.Chem import PandasTools

        mol_df = PandasTools.LoadSDF(
            str(file_path),
            smilesName='SMILES',
            molColName='Molecule',
            includeFingerprints=True
        )

        molecules = pd.DataFrame()
        molecules['compound_id'] = mol_df.get('ID', range(len(mol_df)))
        molecules['smiles'] = mol_df['SMILES']

        return molecules

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def predict_toxicity(input_file, output_dir='results'):
    """Main prediction function."""
    print("=" * 60)
    print("TOXICITY PREDICTION SYSTEM")
    print("=" * 60)

    # Load models
    predictor = load_or_train_models()

    # Load input data
    print(f"\nLoading input file: {input_file}")
    molecules = load_input_file(input_file)
    print(f"Loaded {len(molecules)} molecules")

    # Generate features
    print("Generating molecular features...")
    feature_gen = MolecularFeatureGenerator()
    X = feature_gen.generate_features(
        molecules['smiles'].tolist(),
        use_morgan=True,
        use_maccs=True,
        use_descriptors=True
    )

    # Make predictions
    print("Making toxicity predictions...")
    predictions = predictor.predict(X)

    # Process results
    results = molecules.copy()
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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    csv_path = os.path.join(output_dir, 'toxicity_predictions.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Generate summary report
    generate_summary_report(results, output_dir)

    # Display results
    display_results(results)

    return results


def generate_summary_report(results_df, output_dir):
    """Generate a summary report."""
    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TOXICITY PREDICTION SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Compounds Analyzed: {len(results_df)}\n\n")

        # Target-wise summary
        f.write("TARGET-WISE TOXICITY SUMMARY:\n")
        f.write("-" * 40 + "\n")

        for target in target_columns:
            tox_col = f'{target}_toxicity'
            if tox_col in results_df.columns:
                toxicity_counts = results_df[tox_col].value_counts()
                f.write(f"\n{target}:\n")
                for toxicity, count in toxicity_counts.items():
                    f.write(f"  {toxicity}: {count} compounds\n")

        # Compound-wise summary
        f.write("\n\nCOMPOUND-WISE SUMMARY:\n")
        f.write("-" * 40 + "\n")

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

            f.write(f"\n{row['compound_id']}:\n")
            f.write(f"  Toxic targets: {toxic_count}\n")
            f.write(f"  Moderate targets: {moderate_count}\n")
            f.write(f"  Non-toxic targets: {non_toxic_count}\n")

            if toxic_count > 0:
                f.write("  ⚠️  OVERALL: TOXIC\n")
            elif moderate_count > 0:
                f.write("  ⚠️  OVERALL: MODERATE TOXICITY\n")
            else:
                f.write("  ✅ OVERALL: SAFE\n")

    print(f"Summary report saved to: {report_path}")


def display_results(results_df):
    """Display results in console."""
    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    for idx, row in results_df.iterrows():
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
                    print(f"  🔴 {target}: {toxicity} ({probability:.3f})")
                elif toxicity == 'Moderate':
                    moderate_count += 1
                    print(f"  🟡 {target}: {toxicity} ({probability:.3f})")
                else:
                    non_toxic_count += 1
                    print(f"  🟢 {target}: {toxicity} ({probability:.3f})")

        print(f"\nOverall: {toxic_count} Toxic, {moderate_count} Moderate, {non_toxic_count} Non-toxic")

        if toxic_count > 0:
            print("⚠️  WARNING: This compound shows toxic properties!")
        elif moderate_count > 0:
            print("⚠️  CAUTION: This compound shows moderate toxicity.")
        else:
            print("✅ SAFE: This compound appears non-toxic.")

        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Toxicity Prediction System')
    parser.add_argument('input_file', help='Input file (CSV or SDF)')
    parser.add_argument('--output_dir', default='results', help='Output directory')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return

    try:
        results = predict_toxicity(args.input_file, args.output_dir)
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_dir}/")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()