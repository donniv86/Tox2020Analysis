#!/usr/bin/env python3
"""
Comprehensive EDA (Exploratory Data Analysis) for Tox21 .sdf files
Analyzes molecular properties, target distributions, and data quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import warnings
warnings.filterwarnings('ignore')

# Import our existing data processing module
from src.data_processing import Tox21DataLoader

class Tox21EDA:
    """
    Comprehensive EDA analysis for Tox21 dataset
    """

    def __init__(self, sdf_path: str = "data/tox21_10k_data_all.sdf"):
        """
        Initialize EDA analyzer

        Args:
            sdf_path: Path to the SDF file
        """
        self.sdf_path = sdf_path
        self.data_loader = Tox21DataLoader(sdf_path)
        self.data = None
        self.raw_data = None

    def load_and_analyze(self):
        """Load data and perform comprehensive analysis"""
        print("🔬 Starting Tox21 EDA Analysis...")
        print("=" * 50)

        # Load raw SDF data first for detailed analysis
        self._load_raw_data()

        # Load processed data using existing loader
        self.data = self.data_loader.load_data()

        # Perform comprehensive analysis
        self._analyze_basic_statistics()
        self._analyze_molecular_properties()
        self._analyze_target_distributions()
        self._analyze_data_quality()
        self._analyze_molecular_diversity()
        self._create_visualizations()

        print("\n✅ EDA Analysis Complete!")

    def _load_raw_data(self):
        """Load raw SDF data for detailed analysis"""
        print("📁 Loading raw SDF data...")

        # Load SDF file with all available properties
        self.raw_data = PandasTools.LoadSDF(
            self.sdf_path,
            smilesName='SMILES',
            molColName='Molecule',
            includeFingerprints=True
        )

        print(f"📊 Loaded {len(self.raw_data)} compounds from SDF file")
        print(f"📋 Available columns: {list(self.raw_data.columns)}")

    def _analyze_basic_statistics(self):
        """Analyze basic dataset statistics"""
        print("\n📈 Basic Dataset Statistics:")
        print("-" * 30)

        compounds = self.data['compounds']
        targets = self.data['targets']

        print(f"Total compounds: {len(compounds):,}")
        print(f"Compounds with target data: {len(targets):,}")
        print(f"Target endpoints: {len(self.data_loader.target_columns)}")

        # Target coverage analysis
        target_coverage = targets[self.data_loader.target_columns].notna().sum()
        print(f"\nTarget coverage:")
        for target, count in target_coverage.items():
            coverage_pct = (count / len(targets)) * 100
            print(f"  {target}: {count:,} compounds ({coverage_pct:.1f}%)")

    def _analyze_molecular_properties(self):
        """Analyze molecular properties and Lipinski's Rule of Five"""
        print("\n🧪 Molecular Properties Analysis:")
        print("-" * 35)

        compounds = self.data['compounds']

        # Calculate additional molecular properties
        molecular_props = []

        for idx, row in compounds.iterrows():
            mol = row['molecule']
            if mol is not None:
                props = {
                    'compound_id': row['compound_id'],
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'rings': Descriptors.RingCount(mol),
                    'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                    'fraction_csp3': Descriptors.FractionCSP3(mol),
                    'slogp': Descriptors.SlogP_VSA1(mol),
                }
                molecular_props.append(props)

        self.molecular_props_df = pd.DataFrame(molecular_props)

        # Basic statistics
        print("Molecular Weight Statistics:")
        print(f"  Mean: {self.molecular_props_df['molecular_weight'].mean():.2f}")
        print(f"  Median: {self.molecular_props_df['molecular_weight'].median():.2f}")
        print(f"  Min: {self.molecular_props_df['molecular_weight'].min():.2f}")
        print(f"  Max: {self.molecular_props_df['molecular_weight'].max():.2f}")

        print(f"\nLogP Statistics:")
        print(f"  Mean: {self.molecular_props_df['logp'].mean():.2f}")
        print(f"  Median: {self.molecular_props_df['logp'].median():.2f}")
        print(f"  Min: {self.molecular_props_df['logp'].min():.2f}")
        print(f"  Max: {self.molecular_props_df['logp'].max():.2f}")

        # Lipinski's Rule of Five analysis
        self._analyze_lipinski_rules()

    def _analyze_lipinski_rules(self):
        """Analyze compliance with Lipinski's Rule of Five"""
        print("\n📋 Lipinski's Rule of Five Analysis:")
        print("-" * 35)

        df = self.molecular_props_df

        # Check each rule
        rule1 = df['molecular_weight'] <= 500  # MW ≤ 500
        rule2 = df['logp'] <= 5  # LogP ≤ 5
        rule3 = df['hbd'] <= 5  # HBD ≤ 5
        rule4 = df['hba'] <= 10  # HBA ≤ 10

        compliance = rule1 & rule2 & rule3 & rule4

        print(f"Compounds following all 4 rules: {compliance.sum():,} ({compliance.mean()*100:.1f}%)")
        print(f"Compounds following 3+ rules: {(rule1 + rule2 + rule3 + rule4 >= 3).sum():,}")
        print(f"Compounds following 2+ rules: {(rule1 + rule2 + rule3 + rule4 >= 2).sum():,}")

        # Individual rule compliance
        print(f"\nIndividual rule compliance:")
        print(f"  MW ≤ 500: {rule1.sum():,} ({rule1.mean()*100:.1f}%)")
        print(f"  LogP ≤ 5: {rule2.sum():,} ({rule2.mean()*100:.1f}%)")
        print(f"  HBD ≤ 5: {rule3.sum():,} ({rule3.mean()*100:.1f}%)")
        print(f"  HBA ≤ 10: {rule4.sum():,} ({rule4.mean()*100:.1f}%)")

    def _analyze_target_distributions(self):
        """Analyze target distributions and correlations"""
        print("\n🎯 Target Distribution Analysis:")
        print("-" * 32)

        targets = self.data['targets']

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                positive_count = target_data.sum()
                negative_count = (target_data == 0).sum()
                positive_ratio = positive_count / len(target_data)

                print(f"{target}:")
                print(f"  Total samples: {len(target_data):,}")
                print(f"  Positive: {positive_count:,} ({positive_ratio*100:.1f}%)")
                print(f"  Negative: {negative_count:,} ({(1-positive_ratio)*100:.1f}%)")
                print()

    def _analyze_data_quality(self):
        """Analyze data quality and missing values"""
        print("\n🔍 Data Quality Analysis:")
        print("-" * 25)

        # Missing values in raw data
        missing_data = self.raw_data.isnull().sum()
        missing_pct = (missing_data / len(self.raw_data)) * 100

        print("Missing values in raw SDF data:")
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count:,} ({missing_pct[col]:.1f}%)")

        # SMILES validity
        valid_smiles = 0
        total_smiles = len(self.raw_data)

        for smiles in self.raw_data['SMILES']:
            if pd.notna(smiles) and Chem.MolFromSmiles(smiles) is not None:
                valid_smiles += 1

        print(f"\nSMILES validity: {valid_smiles:,}/{total_smiles:,} ({valid_smiles/total_smiles*100:.1f}%)")

    def _analyze_molecular_diversity(self):
        """Analyze molecular diversity using fingerprints"""
        print("\n🌐 Molecular Diversity Analysis:")
        print("-" * 32)

        # Calculate Morgan fingerprints for diversity analysis
        fingerprints = []
        valid_mols = []

        for mol in self.data['compounds']['molecule']:
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
                valid_mols.append(mol)

        print(f"Valid molecules for fingerprinting: {len(fingerprints):,}")

        if len(fingerprints) > 1000:
            # Sample for diversity analysis to avoid memory issues
            sample_size = 1000
            indices = np.random.choice(len(fingerprints), sample_size, replace=False)
            fingerprints_sample = [fingerprints[i] for i in indices]
            print(f"Using random sample of {sample_size} molecules for diversity analysis")
        else:
            fingerprints_sample = fingerprints

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fingerprints_sample)):
            for j in range(i+1, len(fingerprints_sample)):
                similarity = rdMolDescriptors.FingerprintSimilarity(
                    fingerprints_sample[i],
                    fingerprints_sample[j]
                )
                similarities.append(similarity)

        if similarities:
            similarities = np.array(similarities)
            print(f"Average pairwise similarity: {similarities.mean():.3f}")
            print(f"Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
            print(f"High similarity pairs (≥0.8): {(similarities >= 0.8).sum():,}")

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating Visualizations...")

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create a comprehensive dashboard
        self._create_molecular_properties_plots()
        self._create_target_distribution_plots()
        self._create_lipinski_analysis_plots()
        self._create_correlation_heatmap()

        print("✅ Visualizations created successfully!")

    def _create_molecular_properties_plots(self):
        """Create plots for molecular properties"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Molecular Properties Distribution', fontsize=16, fontweight='bold')

        properties = ['molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds']
        titles = ['Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 'TPSA', 'Rotatable Bonds']

        for i, (prop, title) in enumerate(zip(properties, titles)):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            sns.histplot(self.molecular_props_df[prop], kde=True, ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(prop.replace('_', ' ').title())
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('results/molecular_properties_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_target_distribution_plots(self):
        """Create plots for target distributions"""
        targets = self.data['targets']

        # Target distribution bar plot
        fig, ax = plt.subplots(figsize=(12, 8))

        target_stats = []
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                positive_ratio = target_data.mean()
                target_stats.append({
                    'target': target,
                    'positive_ratio': positive_ratio,
                    'sample_count': len(target_data)
                })

        target_df = pd.DataFrame(target_stats)

        bars = ax.bar(range(len(target_df)), target_df['positive_ratio'])
        ax.set_xlabel('Target Endpoints')
        ax.set_ylabel('Positive Ratio')
        ax.set_title('Target Distribution - Positive Ratio by Endpoint', fontweight='bold')
        ax.set_xticks(range(len(target_df)))
        ax.set_xticklabels(target_df['target'], rotation=45, ha='right')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_lipinski_analysis_plots(self):
        """Create plots for Lipinski's Rule of Five analysis"""
        df = self.molecular_props_df

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Lipinski's Rule of Five Analysis", fontsize=16, fontweight='bold')

        # Rule 1: MW ≤ 500
        ax1 = axes[0, 0]
        sns.histplot(df['molecular_weight'], bins=50, ax=ax1)
        ax1.axvline(x=500, color='red', linestyle='--', label='MW ≤ 500')
        ax1.set_title('Molecular Weight Distribution')
        ax1.set_xlabel('Molecular Weight')
        ax1.legend()

        # Rule 2: LogP ≤ 5
        ax2 = axes[0, 1]
        sns.histplot(df['logp'], bins=50, ax=ax2)
        ax2.axvline(x=5, color='red', linestyle='--', label='LogP ≤ 5')
        ax2.set_title('LogP Distribution')
        ax2.set_xlabel('LogP')
        ax2.legend()

        # Rule 3: HBD ≤ 5
        ax3 = axes[1, 0]
        sns.histplot(df['hbd'], bins=20, ax=ax3)
        ax3.axvline(x=5, color='red', linestyle='--', label='HBD ≤ 5')
        ax3.set_title('H-Bond Donors Distribution')
        ax3.set_xlabel('H-Bond Donors')
        ax3.legend()

        # Rule 4: HBA ≤ 10
        ax4 = axes[1, 1]
        sns.histplot(df['hba'], bins=30, ax=ax4)
        ax4.axvline(x=10, color='red', linestyle='--', label='HBA ≤ 10')
        ax4.set_title('H-Bond Acceptors Distribution')
        ax4.set_xlabel('H-Bond Acceptors')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('results/lipinski_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_correlation_heatmap(self):
        """Create correlation heatmap for molecular properties"""
        # Select numeric columns for correlation
        numeric_cols = ['molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds',
                       'aromatic_rings', 'rings', 'heavy_atoms', 'fraction_csp3']

        corr_matrix = self.molecular_props_df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Molecular Properties Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('results/molecular_properties_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Generate a comprehensive EDA report"""
        print("\n📄 Generating EDA Report...")

        report = f"""
# Tox21 Dataset EDA Report

## Dataset Overview
- **Total compounds**: {len(self.data['compounds']):,}
- **Target endpoints**: {len(self.data_loader.target_columns)}
- **Data source**: {self.sdf_path}

## Key Findings

### Molecular Properties
- Average molecular weight: {self.molecular_props_df['molecular_weight'].mean():.2f}
- Average LogP: {self.molecular_props_df['logp'].mean():.2f}
- Lipinski compliance: {(self.molecular_props_df['molecular_weight'] <= 500).mean()*100:.1f}%

### Target Distribution
"""

        targets = self.data['targets']
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                positive_ratio = target_data.mean()
                report += f"- {target}: {positive_ratio*100:.1f}% positive\n"

        report += f"""
## Data Quality
- Valid SMILES: {len(self.data['compounds'])}/{len(self.raw_data)} compounds
- Missing target data: {(targets[self.data_loader.target_columns].isnull().sum().sum()):,} values

## Visualizations Generated
- Molecular properties distribution
- Target distribution analysis
- Lipinski's Rule of Five analysis
- Molecular properties correlation matrix

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open('results/eda_report.md', 'w') as f:
            f.write(report)

        print("✅ EDA Report generated: results/eda_report.md")


def main():
    """Main function to run the EDA analysis"""
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)

    # Initialize and run EDA
    eda = Tox21EDA()
    eda.load_and_analyze()
    eda.generate_report()

    print("\n🎉 EDA Analysis Complete!")
    print("📁 Check the 'results/' directory for generated plots and report")


if __name__ == "__main__":
    main()