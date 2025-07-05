#!/usr/bin/env python3
"""
Test SMILES molecules from CSV file using trained Tox21 models.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.feature_selector import FeatureSelector
from src.comprehensive_descriptors import ComprehensiveDescriptorGenerator

class MoleculeTester:
    """Test molecules using trained Tox21 models."""

    def __init__(self, models_dir='results/models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_selectors = {}
        self.scalers = {}
        self.targets = []
        self.load_models()

    def load_models(self):
        """Load all trained models and their associated components."""
        print("Loading trained models...")

        if not self.models_dir.exists():
            print(f"Models directory {self.models_dir} not found!")
            return

        # Find all model files
        model_files = list(self.models_dir.glob('*_model.pkl'))

        for model_file in model_files:
            target = model_file.stem.replace('_model', '')
            # Remove '_best' from target name for file matching
            base_target = target.replace('_best', '')
            print(f"Loading model for target: {target}")

            try:
                # Load model
                self.models[target] = joblib.load(model_file)

                # Load feature selector
                selector_file = self.models_dir / f"{base_target}_feature_selector.pkl"
                print(f"  Looking for selector: {selector_file}")
                if selector_file.exists():
                    print(f"  Loading selector for {target}")
                    self.feature_selectors[target] = joblib.load(selector_file)
                else:
                    print(f"  Selector file not found: {selector_file}")

                # Load scaler
                scaler_file = self.models_dir / f"{base_target}_scaler.pkl"
                if scaler_file.exists():
                    self.scalers[target] = joblib.load(scaler_file)

                self.targets.append(target)

            except Exception as e:
                print(f"Error loading model for {target}: {e}")

        print(f"Loaded {len(self.targets)} models: {self.targets}")

    def load_test_molecules(self, csv_file):
        """Load SMILES molecules from CSV file."""
        print(f"Loading molecules from {csv_file}...")

        try:
            # Try different separators
            df = pd.read_csv(csv_file, sep=';')

            # Check if Smiles column exists
            if 'Smiles' not in df.columns:
                print("Available columns:", df.columns.tolist())
                raise ValueError("'Smiles' column not found in CSV")

            # Clean SMILES
            df = df.dropna(subset=['Smiles'])
            df['Smiles'] = df['Smiles'].astype(str).str.strip()

            # Remove empty SMILES
            df = df[df['Smiles'] != '']
            df = df[df['Smiles'] != 'nan']

            print(f"Loaded {len(df)} molecules with valid SMILES")
            return df

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def generate_descriptors(self, smiles_list):
        """Generate descriptors for SMILES molecules."""
        print("Generating molecular descriptors...")

        try:
            generator = ComprehensiveDescriptorGenerator()
            descriptors = generator.generate_all_descriptors(smiles_list)

            if descriptors is not None:
                print(f"Generated descriptors for {len(descriptors)} molecules")
                # Convert to DataFrame for compatibility
                descriptors_df = pd.DataFrame(descriptors)
                return descriptors_df
            else:
                print("Failed to generate descriptors")
                return None

        except Exception as e:
            print(f"Error generating descriptors: {e}")
            return None

    def predict_toxicity(self, descriptors_df):
        """Predict toxicity for all targets using loaded models."""
        if not self.targets:
            print("No models loaded!")
            return None

        results = {}

        for target in self.targets:
            print(f"\nPredicting for target: {target}")

            try:
                # Get features for this target
                if target in self.feature_selectors:
                    selector = self.feature_selectors[target]

                    # The FeatureSelector class doesn't have a transform method
                    # We need to manually select the features using the saved feature_names_
                    if hasattr(selector, 'feature_names_') and selector.feature_names_ is not None:
                        # Convert descriptors_df to DataFrame if it's not already
                        if not isinstance(descriptors_df, pd.DataFrame):
                            # Create DataFrame with generic column names
                            descriptors_df = pd.DataFrame(descriptors_df,
                                                        columns=[f'feature_{i}' for i in range(descriptors_df.shape[1])])

                        # Get the selected feature names
                        selected_feature_names = selector.feature_names_

                        # Check if we have the required features
                        missing_features = [f for f in selected_feature_names if f not in descriptors_df.columns]
                        if missing_features:
                            print(f"  Warning: {len(missing_features)} features missing for {target}, filling with 0.")
                            for f in missing_features:
                                descriptors_df[f] = 0

                        # Select the features in the correct order
                        selected_features = descriptors_df[selected_feature_names].values
                        print(f"  Selected {selected_features.shape[1]} features for {target}")
                    else:
                        print(f"  Warning: No feature names found for {target}, using all features")
                        selected_features = descriptors_df.values if isinstance(descriptors_df, np.ndarray) else descriptors_df.values
                else:
                    print(f"  Warning: No feature selector found for {target}, using all features")
                    selected_features = descriptors_df.values if isinstance(descriptors_df, np.ndarray) else descriptors_df.values

                # Scale features if scaler exists
                if target in self.scalers:
                    selected_features = self.scalers[target].transform(selected_features)

                # Make prediction
                model = self.models[target]
                predictions = model.predict(selected_features)
                probabilities = model.predict_proba(selected_features)

                results[target] = {
                    'predictions': predictions,
                    'probabilities': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                }

                # Count predictions
                positive_count = np.sum(predictions == 1)
                total_count = len(predictions)
                print(f"  Positive predictions: {positive_count}/{total_count} ({positive_count/total_count*100:.1f}%)")

            except Exception as e:
                print(f"  Error predicting for {target}: {e}")
                results[target] = None

        return results

    def create_results_dataframe(self, original_df, predictions_dict):
        """Create a results dataframe with predictions."""
        results_df = original_df.copy()

        for target, pred_data in predictions_dict.items():
            if pred_data is not None:
                results_df[f'{target}_prediction'] = pred_data['predictions']
                results_df[f'{target}_probability'] = pred_data['probabilities']

        return results_df

    def save_results(self, results_df, output_file):
        """Save results to CSV file."""
        try:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_summary(self, results_df):
        """Print a summary of predictions."""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)

        # Count total molecules
        total_molecules = len(results_df)
        print(f"Total molecules tested: {total_molecules}")

        # Summary for each target
        for target in self.targets:
            pred_col = f'{target}_prediction'
            prob_col = f'{target}_probability'

            if pred_col in results_df.columns:
                predictions = results_df[pred_col]
                probabilities = results_df[prob_col]

                positive_count = np.sum(predictions == 1)
                avg_prob = np.mean(probabilities)

                print(f"\n{target}:")
                print(f"  Positive predictions: {positive_count}/{total_molecules} ({positive_count/total_molecules*100:.1f}%)")
                print(f"  Average probability: {avg_prob:.3f}")

                # Show top 5 most toxic molecules
                if positive_count > 0:
                    toxic_molecules = results_df[predictions == 1].nlargest(5, prob_col)
                    print(f"  Top 5 most toxic molecules:")
                    for idx, row in toxic_molecules.iterrows():
                        print(f"    {row['Smiles'][:50]}... (prob: {row[prob_col]:.3f})")

def main():
    """Main function to test molecules."""
    print("Tox21 Molecule Testing Pipeline")
    print("="*50)

    # Initialize tester
    tester = MoleculeTester()

    if not tester.targets:
        print("No models found! Please run the training pipeline first.")
        return

    # Load test molecules
    csv_file = "test_folder/jiang.csv"
    if not os.path.exists(csv_file):
        print(f"Test file {csv_file} not found!")
        return

    df = tester.load_test_molecules(csv_file)
    if df is None or len(df) == 0:
        print("No valid molecules found!")
        return

    # Generate descriptors
    descriptors = tester.generate_descriptors(df['Smiles'].tolist())
    if descriptors is None:
        print("Failed to generate descriptors!")
        return

    # Make predictions
    predictions = tester.predict_toxicity(descriptors)
    if predictions is None:
        print("Failed to make predictions!")
        return

    # Create results dataframe
    results_df = tester.create_results_dataframe(df, predictions)

    # Save results
    output_file = "test_folder/toxicity_predictions.csv"
    tester.save_results(results_df, output_file)

    # Print summary
    tester.print_summary(results_df)

    print(f"\nTesting completed! Results saved to: {output_file}")

if __name__ == "__main__":
    main()