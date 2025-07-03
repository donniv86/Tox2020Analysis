#!/usr/bin/env python3
"""
Comprehensive Toxicity Prediction System
Takes molecular files (SMILES CSV, SDF, MOL) and generates detailed toxicity reports.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_processing import Tox21DataLoader
from feature_engineering import MolecularFeatureGenerator
from models import ToxicityPredictor
from evaluation import ModelEvaluator

# Chemistry imports
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import mols2grid

# Report generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from PIL import Image as PILImage


class ToxicityPredictorSystem:
    """Complete toxicity prediction system with reporting capabilities."""

    def __init__(self, model_path: str = None):
        self.target_columns = [
            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

        self.target_descriptions = {
            'NR-Aromatase': 'Nuclear Receptor Aromatase - Estrogen synthesis',
            'NR-AR': 'Nuclear Receptor Androgen Receptor - Male hormone signaling',
            'NR-AR-LBD': 'Nuclear Receptor Androgen Receptor Ligand Binding Domain',
            'NR-ER': 'Nuclear Receptor Estrogen Receptor - Female hormone signaling',
            'NR-ER-LBD': 'Nuclear Receptor Estrogen Receptor Ligand Binding Domain',
            'NR-PPAR-gamma': 'Nuclear Receptor PPAR-gamma - Metabolism regulation',
            'NR-AhR': 'Nuclear Receptor Aryl Hydrocarbon Receptor - Xenobiotic response',
            'SR-ARE': 'Stress Response Antioxidant Response Element - Oxidative stress',
            'SR-ATAD5': 'Stress Response ATAD5 - DNA replication stress',
            'SR-HSE': 'Stress Response Heat Shock Element - Heat shock response',
            'SR-MMP': 'Stress Response Mitochondrial Membrane Potential - Mitochondrial toxicity',
            'SR-p53': 'Stress Response p53 - DNA damage response'
        }

        # Initialize components
        self.feature_gen = MolecularFeatureGenerator()
        self.predictor = ToxicityPredictor('random_forest')

        # Load pre-trained models if available
        if model_path and os.path.exists(model_path):
            self.predictor.load_models(model_path)
            print(f"Loaded pre-trained models from {model_path}")
        else:
            print("No pre-trained models found. Please train models first.")

    def load_molecules(self, file_path: str) -> pd.DataFrame:
        """Load molecules from various file formats."""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() == '.sdf':
            return self._load_sdf(file_path)
        elif file_path.suffix.lower() == '.mol':
            return self._load_mol(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load molecules from CSV file."""
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

        # Validate SMILES
        valid_molecules = []
        for idx, row in molecules.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                valid_molecules.append(row)

        return pd.DataFrame(valid_molecules)

    def _load_sdf(self, file_path: Path) -> pd.DataFrame:
        """Load molecules from SDF file."""
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

    def _load_mol(self, file_path: Path) -> pd.DataFrame:
        """Load single molecule from MOL file."""
        mol = Chem.MolFromMolFile(str(file_path))
        if mol is None:
            raise ValueError("Invalid MOL file")

        molecules = pd.DataFrame()
        molecules['compound_id'] = [file_path.stem]
        molecules['smiles'] = [Chem.MolToSmiles(mol)]

        return molecules

    def predict_toxicity(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """Predict toxicity for all molecules."""
        if not hasattr(self.predictor, 'models') or not self.predictor.models:
            raise ValueError("No trained models available. Please train models first.")

        # Generate features
        X = self.feature_gen.generate_features(
            molecules_df['smiles'].tolist(),
            use_morgan=True,
            use_maccs=True,
            use_descriptors=True
        )

        # Make predictions
        predictions = self.predictor.predict(X)

        # Add compound information
        results = molecules_df.copy()
        for target in self.target_columns:
            if target in predictions.columns:
                results[f'{target}_probability'] = predictions[target]
                results[f'{target}_prediction'] = (predictions[target] > 0.5).astype(int)
                results[f'{target}_toxicity'] = self._classify_toxicity(predictions[target])

        return results

    def _classify_toxicity(self, probabilities: np.ndarray) -> List[str]:
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

    def generate_molecular_images(self, molecules_df: pd.DataFrame, output_dir: str) -> List[str]:
        """Generate molecular structure images."""
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []

        for idx, row in molecules_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                # Generate 2D structure
                img = Draw.MolToImage(mol, size=(300, 300))
                img_path = os.path.join(output_dir, f"molecule_{row['compound_id']}.png")
                img.save(img_path)
                image_paths.append(img_path)
            else:
                image_paths.append(None)

        return image_paths

    def create_toxicity_summary(self, results_df: pd.DataFrame) -> Dict:
        """Create summary statistics for toxicity predictions."""
        summary = {
            'total_compounds': len(results_df),
            'targets': {}
        }

        for target in self.target_columns:
            prob_col = f'{target}_probability'
            tox_col = f'{target}_toxicity'

            if prob_col in results_df.columns:
                target_data = results_df[prob_col].dropna()
                toxicity_counts = results_df[tox_col].value_counts()

                summary['targets'][target] = {
                    'description': self.target_descriptions[target],
                    'average_probability': float(target_data.mean()),
                    'max_probability': float(target_data.max()),
                    'min_probability': float(target_data.min()),
                    'toxicity_distribution': {
                        'Non-toxic': int(toxicity_counts.get('Non-toxic', 0)),
                        'Moderate': int(toxicity_counts.get('Moderate', 0)),
                        'Toxic': int(toxicity_counts.get('Toxic', 0))
                    }
                }

        return summary

    def generate_csv_report(self, results_df: pd.DataFrame, output_path: str):
        """Generate CSV report with all predictions."""
        # Select relevant columns
        report_columns = ['compound_id', 'smiles']
        for target in self.target_columns:
            report_columns.extend([
                f'{target}_probability',
                f'{target}_prediction',
                f'{target}_toxicity'
            ])

        # Filter available columns
        available_columns = [col for col in report_columns if col in results_df.columns]
        report_df = results_df[available_columns].copy()

        # Save to CSV
        report_df.to_csv(output_path, index=False)
        print(f"CSV report saved to: {output_path}")

    def generate_pdf_report(self, results_df: pd.DataFrame, summary: Dict,
                          image_paths: List[str], output_path: str):
        """Generate comprehensive PDF report."""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Toxicity Prediction Report", title_style))
        story.append(Spacer(1, 20))

        # Summary section
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 12))

        summary_text = f"""
        This report contains toxicity predictions for {summary['total_compounds']} compounds
        across {len(summary['targets'])} Tox21 endpoints. The predictions are based on
        machine learning models trained on the Tox21 dataset.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Overall statistics
        story.append(Paragraph("Overall Statistics", styles['Heading3']))
        story.append(Spacer(1, 12))

        # Create summary table
        summary_data = [['Metric', 'Value']]
        summary_data.append(['Total Compounds', str(summary['total_compounds'])])
        summary_data.append(['Toxicity Endpoints', str(len(summary['targets']))])

        # Calculate overall toxic compounds
        toxic_counts = []
        for target_data in summary['targets'].values():
            toxic_counts.append(target_data['toxicity_distribution']['Toxic'])

        summary_data.append(['Average Toxic Compounds per Target', f"{np.mean(toxic_counts):.1f}"])

        summary_table = Table(summary_data, colWidths=[2*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Target-wise results
        story.append(Paragraph("Target-wise Results", styles['Heading2']))
        story.append(Spacer(1, 12))

        for target, target_data in summary['targets'].items():
            story.append(Paragraph(f"{target}", styles['Heading3']))
            story.append(Paragraph(target_data['description'], styles['Normal']))
            story.append(Spacer(1, 6))

            # Target statistics table
            target_stats = [
                ['Metric', 'Value'],
                ['Average Probability', f"{target_data['average_probability']:.3f}"],
                ['Max Probability', f"{target_data['max_probability']:.3f}"],
                ['Min Probability', f"{target_data['min_probability']:.3f}"],
                ['Non-toxic Compounds', str(target_data['toxicity_distribution']['Non-toxic'])],
                ['Moderate Compounds', str(target_data['toxicity_distribution']['Moderate'])],
                ['Toxic Compounds', str(target_data['toxicity_distribution']['Toxic'])]
            ]

            target_table = Table(target_stats, colWidths=[1.5*inch, 1*inch])
            target_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(target_table)
            story.append(Spacer(1, 15))

        # Compound details
        story.append(Paragraph("Compound Details", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Create compound table
        compound_data = [['Compound ID', 'SMILES', 'Toxicity Summary']]

        for idx, row in results_df.iterrows():
            # Count toxic predictions
            toxic_count = 0
            moderate_count = 0
            for target in self.target_columns:
                tox_col = f'{target}_toxicity'
                if tox_col in row and pd.notna(row[tox_col]):
                    if row[tox_col] == 'Toxic':
                        toxic_count += 1
                    elif row[tox_col] == 'Moderate':
                        moderate_count += 1

            # Create summary
            if toxic_count > 0:
                summary_text = f"Toxic: {toxic_count}, Moderate: {moderate_count}"
            elif moderate_count > 0:
                summary_text = f"Moderate: {moderate_count}"
            else:
                summary_text = "Non-toxic"

            compound_data.append([
                str(row['compound_id']),
                row['smiles'][:50] + "..." if len(row['smiles']) > 50 else row['smiles'],
                summary_text
            ])

        # Limit table size for PDF
        if len(compound_data) > 20:
            compound_data = compound_data[:20]
            compound_data.append(['...', '...', '...'])

        compound_table = Table(compound_data, colWidths=[1*inch, 2*inch, 1.5*inch])
        compound_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(compound_table)

        # Build PDF
        doc.build(story)
        print(f"PDF report saved to: {output_path}")

    def create_visualization_report(self, results_df: pd.DataFrame, output_dir: str):
        """Create visualization plots for the results."""
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Toxicity distribution across targets
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        for i, target in enumerate(self.target_columns):
            tox_col = f'{target}_toxicity'
            if tox_col in results_df.columns:
                toxicity_counts = results_df[tox_col].value_counts()
                colors_map = {'Non-toxic': 'green', 'Moderate': 'orange', 'Toxic': 'red'}
                colors_list = [colors_map.get(x, 'blue') for x in toxicity_counts.index]

                axes[i].pie(toxicity_counts.values, labels=toxicity_counts.index,
                           colors=colors_list, autopct='%1.1f%%')
                axes[i].set_title(target, fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'toxicity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Probability heatmap
        prob_columns = [f'{target}_probability' for target in self.target_columns
                       if f'{target}_probability' in results_df.columns]

        if prob_columns:
            prob_data = results_df[prob_columns].values
            plt.figure(figsize=(12, 8))
            sns.heatmap(prob_data, cmap='RdYlGn_r', center=0.5,
                       xticklabels=[col.replace('_probability', '') for col in prob_columns],
                       yticklabels=results_df['compound_id'])
            plt.title('Toxicity Probability Heatmap')
            plt.xlabel('Toxicity Targets')
            plt.ylabel('Compounds')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'probability_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Overall toxicity summary
        toxic_summary = []
        for idx, row in results_df.iterrows():
            toxic_count = 0
            moderate_count = 0
            for target in self.target_columns:
                tox_col = f'{target}_toxicity'
                if tox_col in row and pd.notna(row[tox_col]):
                    if row[tox_col] == 'Toxic':
                        toxic_count += 1
                    elif row[tox_col] == 'Moderate':
                        moderate_count += 1

            if toxic_count > 0:
                toxic_summary.append('Toxic')
            elif moderate_count > 0:
                toxic_summary.append('Moderate')
            else:
                toxic_summary.append('Non-toxic')

        plt.figure(figsize=(10, 6))
        summary_counts = pd.Series(toxic_summary).value_counts()
        colors_map = {'Non-toxic': 'green', 'Moderate': 'orange', 'Toxic': 'red'}
        colors_list = [colors_map.get(x, 'blue') for x in summary_counts.index]

        plt.bar(summary_counts.index, summary_counts.values, color=colors_list)
        plt.title('Overall Toxicity Summary')
        plt.ylabel('Number of Compounds')
        plt.xlabel('Toxicity Classification')

        # Add value labels on bars
        for i, v in enumerate(summary_counts.values):
            plt.text(i, v + 0.1, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_toxicity_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to: {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Toxicity Prediction System')
    parser.add_argument('input_file', help='Input file (CSV, SDF, or MOL)')
    parser.add_argument('--output_dir', default='toxicity_results', help='Output directory')
    parser.add_argument('--model_path', help='Path to pre-trained models')
    parser.add_argument('--format', choices=['csv', 'pdf', 'both'], default='both',
                       help='Output format')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize system
    print("Initializing Toxicity Prediction System...")
    system = ToxicityPredictorSystem(args.model_path)

    # Load molecules
    print(f"Loading molecules from {args.input_file}...")
    molecules = system.load_molecules(args.input_file)
    print(f"Loaded {len(molecules)} molecules")

    # Predict toxicity
    print("Predicting toxicity...")
    results = system.predict_toxicity(molecules)

    # Generate summary
    print("Generating summary...")
    summary = system.create_toxicity_summary(results)

    # Generate molecular images
    print("Generating molecular images...")
    image_dir = os.path.join(args.output_dir, 'molecular_images')
    image_paths = system.generate_molecular_images(molecules, image_dir)

    # Generate reports
    if args.format in ['csv', 'both']:
        csv_path = os.path.join(args.output_dir, 'toxicity_predictions.csv')
        system.generate_csv_report(results, csv_path)

    if args.format in ['pdf', 'both']:
        pdf_path = os.path.join(args.output_dir, 'toxicity_report.pdf')
        system.generate_pdf_report(results, summary, image_paths, pdf_path)

    # Generate visualizations
    print("Generating visualizations...")
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    system.create_visualization_report(results, viz_dir)

    print(f"\nResults saved to: {args.output_dir}")
    print("Files generated:")
    print(f"  - toxicity_predictions.csv (detailed predictions)")
    print(f"  - toxicity_report.pdf (comprehensive report)")
    print(f"  - molecular_images/ (structure images)")
    print(f"  - visualizations/ (plots and charts)")


if __name__ == "__main__":
    main()