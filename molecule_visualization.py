#!/usr/bin/env python3
"""
Focused molecular visualization for Tox21 toxicity predictions.
Shows RDKit molecule images with red/green indicators for pass/fail status.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

class MoleculeVisualizer:
    """Visualize molecules with toxicity predictions using RDKit."""

    def __init__(self, results_file='test_folder/toxicity_predictions.csv'):
        self.results_file = results_file
        self.df = None
        self.targets = []
        self.load_results()

    def load_results(self):
        """Load prediction results from CSV."""
        if not os.path.exists(self.results_file):
            print(f"Results file {self.results_file} not found!")
            return

        self.df = pd.read_csv(self.results_file)
        print(f"Loaded {len(self.df)} molecules with predictions")

        # Extract target names from column names
        self.targets = []
        for col in self.df.columns:
            if col.endswith('_prediction'):
                target = col.replace('_prediction', '')
                self.targets.append(target)

        print(f"Found {len(self.targets)} targets: {self.targets}")

    def create_molecule_image(self, smiles, mol_id, predictions, probabilities):
        """Create a single molecule visualization with toxicity indicators."""
        try:
            # Create RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(f'Molecule {mol_id}', fontsize=14, fontweight='bold')

            # 1. Molecule structure
            ax1.set_title('Molecular Structure', fontsize=12)
            img = Draw.MolToImage(mol, size=(400, 300))
            ax1.imshow(img)
            ax1.axis('off')

            # 2. Toxicity predictions
            ax2.set_title('Toxicity Predictions', fontsize=12)
            ax2.axis('off')

            # Create prediction grid
            n_targets = len(self.targets)
            cols = 3
            rows = (n_targets + cols - 1) // cols

            for i, target in enumerate(self.targets):
                row = i // cols
                col = i % cols

                # Get prediction and probability
                pred = predictions.get(target, 0)
                prob = probabilities.get(target, 0.0)

                # Color based on prediction (green=pass, red=fail)
                color = 'green' if pred == 0 else 'red'
                status = 'PASS' if pred == 0 else 'FAIL'

                # Create text box
                text = f'{target}\n{status}\n{prob:.3f}'
                ax2.text(0.1 + col * 0.3, 0.9 - row * 0.15, text,
                        transform=ax2.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=color, alpha=0.7),
                        ha='center', va='center', fontweight='bold',
                        color='white')

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error creating image for molecule {mol_id}: {e}")
            return None

    def create_summary_grid(self, pdf_pages):
        """Create a summary grid showing all molecules with their status."""
        # Calculate average toxicity for each molecule
        avg_toxicity = []
        for _, row in self.df.iterrows():
            probs = []
            for target in self.targets:
                prob_col = f'{target}_probability'
                if prob_col in self.df.columns:
                    probs.append(row[prob_col])
            avg_toxicity.append(np.mean(probs))

        self.df['avg_toxicity'] = avg_toxicity

        # Sort by average toxicity (most toxic first)
        sorted_df = self.df.sort_values('avg_toxicity', ascending=False)

        # Create summary grid
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        fig.suptitle('Molecule Toxicity Summary', fontsize=16, fontweight='bold')
        ax.axis('off')

        # Create grid layout
        n_molecules = len(sorted_df)
        cols = 4
        rows = (n_molecules + cols - 1) // cols

        for i, (_, molecule) in enumerate(sorted_df.iterrows()):
            row = i // cols
            col = i % cols

            # Get predictions and probabilities
            predictions = {}
            probabilities = {}
            for target in self.targets:
                pred_col = f'{target}_prediction'
                prob_col = f'{target}_probability'
                if pred_col in self.df.columns and prob_col in self.df.columns:
                    predictions[target] = molecule[pred_col]
                    probabilities[target] = molecule[prob_col]

            # Count passed/failed targets
            passed = sum(1 for p in predictions.values() if p == 0)
            failed = sum(1 for p in predictions.values() if p == 1)
            total = len(predictions)

            # Overall status color
            if failed == 0:
                status_color = 'green'
                status = 'ALL PASS'
            elif passed == 0:
                status_color = 'red'
                status = 'ALL FAIL'
            else:
                status_color = 'orange'
                status = f'{passed}/{total} PASS'

            # Create molecule box
            x = 0.05 + col * 0.225
            y = 0.95 - row * 0.08

            # Background rectangle
            rect = plt.Rectangle((x, y-0.07), 0.2, 0.06,
                               facecolor=status_color, alpha=0.3)
            ax.add_patch(rect)

            # Text
            text = f'Molecule {i+1}\n{status}\nAvg: {molecule["avg_toxicity"]:.3f}'
            ax.text(x + 0.1, y - 0.035, text, ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close()

    def create_individual_molecules(self, pdf_pages, max_molecules=50):
        """Create individual molecule pages with detailed predictions."""
        # Calculate average toxicity
        avg_toxicity = []
        for _, row in self.df.iterrows():
            probs = []
            for target in self.targets:
                prob_col = f'{target}_probability'
                if prob_col in self.df.columns:
                    probs.append(row[prob_col])
            avg_toxicity.append(np.mean(probs))

        self.df['avg_toxicity'] = avg_toxicity

        # Sort by average toxicity and take top molecules
        sorted_df = self.df.sort_values('avg_toxicity', ascending=False).head(max_molecules)

        print(f"Creating detailed visualizations for top {len(sorted_df)} molecules...")

        for i, (_, molecule) in enumerate(sorted_df.iterrows()):
            # Get predictions and probabilities
            predictions = {}
            probabilities = {}
            for target in self.targets:
                pred_col = f'{target}_prediction'
                prob_col = f'{target}_probability'
                if pred_col in self.df.columns and prob_col in self.df.columns:
                    predictions[target] = molecule[pred_col]
                    probabilities[target] = molecule[prob_col]

            # Create molecule image
            fig = self.create_molecule_image(
                molecule['Smiles'],
                i+1,
                predictions,
                probabilities
            )

            if fig is not None:
                pdf_pages.savefig(fig)
                plt.close(fig)
            else:
                print(f"Failed to create image for molecule {i+1}")

    def generate_molecule_report(self, output_pdf='test_folder/molecule_predictions.pdf', max_molecules=50):
        """Generate focused molecular visualization report."""
        if self.df is None:
            print("No data loaded!")
            return

        print(f"Generating focused molecular visualization...")
        print(f"Output will be saved to: {output_pdf}")
        print(f"Showing top {max_molecules} molecules by toxicity")

        with PdfPages(output_pdf) as pdf_pages:
            # 1. Summary grid
            print("Creating summary grid...")
            self.create_summary_grid(pdf_pages)

            # 2. Individual molecules
            print("Creating individual molecule pages...")
            self.create_individual_molecules(pdf_pages, max_molecules)

        print(f"✅ Molecular visualization completed!")
        print(f"📄 PDF Report: {output_pdf}")
        print(f"📊 Molecules visualized: {min(max_molecules, len(self.df))}")

def main():
    """Main function to generate focused molecular visualization."""
    print("=" * 60)
    print("MOLECULAR TOXICITY PREDICTION VISUALIZATION")
    print("=" * 60)

    # Initialize visualizer
    visualizer = MoleculeVisualizer()

    if visualizer.df is None:
        print("❌ Failed to load prediction results!")
        return

    # Generate focused molecular report
    output_file = 'test_folder/molecule_predictions.pdf'
    visualizer.generate_molecule_report(output_file, max_molecules=30)

    print("\n" + "=" * 60)
    print("✅ MOLECULAR VISUALIZATION COMPLETED!")
    print("=" * 60)
    print(f"📄 PDF Report: {output_file}")
    print(f"📊 Molecules Analyzed: {len(visualizer.df)}")
    print(f"🎯 Targets: {len(visualizer.targets)}")
    print("\nThe PDF contains:")
    print("  • Summary grid of all molecules")
    print("  • Individual molecule pages with RDKit structures")
    print("  • Red/Green indicators for pass/fail status")
    print("  • Toxicity probabilities for each target")

if __name__ == "__main__":
    main()