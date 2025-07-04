#!/usr/bin/env python3
"""
Enhanced EDA (Exploratory Data Analysis) for Tox21 .sdf files
Includes fingerprint generation, cluster diversity analysis, and comprehensive molecular diversity metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
warnings.filterwarnings('ignore')

# Import our existing data processing module
from src.data_processing import Tox21DataLoader

class Tox21EDAEnhanced:
    """
    Enhanced EDA analysis for Tox21 dataset with fingerprinting and clustering
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
        self.fingerprints = None
        self.cluster_results = None

    def load_and_analyze(self):
        """Load data and perform comprehensive analysis"""
        print("🔬 Starting Enhanced Tox21 EDA Analysis...")
        print("=" * 60)

        # Load raw SDF data first for detailed analysis
        self._load_raw_data()

        # Load processed data using existing loader
        self.data = self.data_loader.load_data()

        # Perform comprehensive analysis
        self._analyze_basic_statistics()
        self._analyze_target_distributions()
        self._analyze_data_quality()
        self._analyze_scaffold_decomposition()
        self._analyze_molecular_properties()
        self._analyze_data_quality()
        self._generate_fingerprints()
        self._analyze_molecular_diversity()
        self._perform_clustering_analysis()
        self._create_visualizations()

        print("\n✅ Enhanced EDA Analysis Complete!")

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

    def _analyze_target_distributions(self):
        """Analyze target distributions with comprehensive active/inactive analysis"""
        print("\n🎯 Analyzing Target Distributions with Active/Inactive Analysis...")

        targets = self.data['targets']

        # Comprehensive target analysis
        target_analysis = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                positive_count = (target_data == 1).sum()
                negative_count = (target_data == 0).sum()
                total_count = len(target_data)
                positive_ratio = positive_count / total_count

                target_analysis[target] = {
                    'total_compounds': total_count,
                    'active_compounds': positive_count,
                    'inactive_compounds': negative_count,
                    'activity_ratio': positive_ratio,
                    'class_imbalance_ratio': negative_count / positive_count if positive_count > 0 else float('inf'),
                    'data_completeness': total_count / len(self.data['compounds'])
                }

        # Print comprehensive analysis
        print("\n📊 Target Activity Analysis:")
        print("=" * 80)
        print(f"{'Target':<20} {'Active':<8} {'Inactive':<8} {'Ratio':<8} {'Imbalance':<10} {'Completeness':<12}")
        print("-" * 80)

        for target, stats in target_analysis.items():
            print(f"{target:<20} {stats['active_compounds']:<8} {stats['inactive_compounds']:<8} "
                  f"{stats['activity_ratio']:<8.3f} {stats['class_imbalance_ratio']:<10.1f} "
                  f"{stats['data_completeness']:<12.3f}")

        # Identify balanced vs imbalanced targets
        balanced_targets = []
        imbalanced_targets = []

        for target, stats in target_analysis.items():
            if 0.2 <= stats['activity_ratio'] <= 0.8:
                balanced_targets.append(target)
            else:
                imbalanced_targets.append(target)

        print(f"\n🎯 Target Classification:")
        print(f"Balanced targets (20-80% active): {len(balanced_targets)}")
        print(f"Imbalanced targets: {len(imbalanced_targets)}")

        if balanced_targets:
            print(f"Balanced: {', '.join(balanced_targets)}")
        if imbalanced_targets:
            print(f"Imbalanced: {', '.join(imbalanced_targets)}")

        self.target_analysis = target_analysis
        self.balanced_targets = balanced_targets
        self.imbalanced_targets = imbalanced_targets

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

    def _analyze_scaffold_decomposition(self):
        """Analyze chemical scaffold decomposition and diversity per target"""
        print("\n🏗️ Analyzing Chemical Scaffold Decomposition per Target...")

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Generate scaffolds using Bemis-Murcko decomposition
        scaffold_data = []

        for idx, row in compounds.iterrows():
            mol = row['molecule']
            if mol is not None:
                try:
                    # Generate Murcko scaffold
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)

                    # Get scaffold properties
                    scaffold_mw = Descriptors.MolWt(scaffold)
                    scaffold_rings = Descriptors.RingCount(scaffold)
                    scaffold_aromatic_rings = Descriptors.NumAromaticRings(scaffold)

                    scaffold_data.append({
                        'compound_id': row['compound_id'],
                        'scaffold_smiles': scaffold_smiles,
                        'scaffold_mw': scaffold_mw,
                        'scaffold_rings': scaffold_rings,
                        'scaffold_aromatic_rings': scaffold_aromatic_rings,
                        'original_mw': Descriptors.MolWt(mol)
                    })
                except:
                    continue

        self.scaffold_df = pd.DataFrame(scaffold_data)
        self.scaffold_df.set_index('compound_id', inplace=True)

        # Overall scaffold analysis
        unique_scaffolds = self.scaffold_df['scaffold_smiles'].nunique()
        total_compounds = len(self.scaffold_df)
        scaffold_coverage = unique_scaffolds / total_compounds

        print(f"\n📊 Overall Scaffold Analysis:")
        print(f"Total compounds: {total_compounds:,}")
        print(f"Unique scaffolds: {unique_scaffolds:,}")
        print(f"Scaffold diversity ratio: {scaffold_coverage:.3f}")
        print(f"Average compounds per scaffold: {total_compounds/unique_scaffolds:.1f}")

        # Target-specific scaffold analysis
        print(f"\n🎯 Target-Specific Scaffold Analysis:")
        print("=" * 100)

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                valid_indices = target_data.index.intersection(self.scaffold_df.index)
                if len(valid_indices) > 0:
                    target_values = target_data.loc[valid_indices]
                    scaffold_with_target = self.scaffold_df.loc[valid_indices]

                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        active_scaffolds = scaffold_with_target.loc[active_mask, 'scaffold_smiles']
                        inactive_scaffolds = scaffold_with_target.loc[inactive_mask, 'scaffold_smiles']

                        active_unique = active_scaffolds.nunique()
                        inactive_unique = inactive_scaffolds.nunique()
                        total_unique = scaffold_with_target['scaffold_smiles'].nunique()

                        # Find scaffolds unique to active/inactive
                        active_scaffold_set = set(active_scaffolds.unique())
                        inactive_scaffold_set = set(inactive_scaffolds.unique())

                        active_only = len(active_scaffold_set - inactive_scaffold_set)
                        inactive_only = len(inactive_scaffold_set - active_scaffold_set)
                        shared = len(active_scaffold_set & inactive_scaffold_set)

                        print(f"\n🎯 {target}:")
                        print(f"Active compounds: {active_mask.sum()}, Inactive compounds: {inactive_mask.sum()}")
                        print(f"Active unique scaffolds: {active_unique}, Inactive unique scaffolds: {inactive_unique}")
                        print(f"Active-only scaffolds: {active_only}, Inactive-only scaffolds: {inactive_only}")
                        print(f"Shared scaffolds: {shared}")

                        # Scaffold properties by activity
                        print(f"\nScaffold Properties by Activity:")
                        print("-" * 50)
                        print(f"{'Property':<15} {'Active_Mean':<12} {'Inactive_Mean':<14} {'Difference':<12}")
                        print("-" * 50)

                        for prop in ['scaffold_mw', 'scaffold_rings', 'scaffold_aromatic_rings']:
                            active_values = scaffold_with_target.loc[active_mask, prop]
                            inactive_values = scaffold_with_target.loc[inactive_mask, prop]

                            active_mean = active_values.mean()
                            inactive_mean = inactive_values.mean()
                            difference = active_mean - inactive_mean

                            print(f"{prop:<15} {active_mean:<12.2f} {inactive_mean:<14.2f} {difference:<12.2f}")

        # Analyze scaffold distribution by target activity
        self._analyze_scaffold_target_activity()

        print("✅ Scaffold decomposition analysis completed!")

    def _analyze_scaffold_target_activity(self):
        """Analyze scaffold distribution across target activities"""
        print("\n🎯 Analyzing Scaffold Distribution by Target Activity...")

        targets = self.data['targets']
        scaffold_target_analysis = {}

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                # Get compounds with both scaffold and target data
                valid_compounds = target_data.index.intersection(self.scaffold_df.index)

                if len(valid_compounds) > 0:
                    target_values = target_data.loc[valid_compounds]
                    scaffold_with_target = self.scaffold_df.loc[valid_compounds]

                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        active_scaffolds = scaffold_with_target.loc[active_mask, 'scaffold_smiles']
                        inactive_scaffolds = scaffold_with_target.loc[inactive_mask, 'scaffold_smiles']

                        active_unique = active_scaffolds.nunique()
                        inactive_unique = inactive_scaffolds.nunique()
                        total_unique = scaffold_with_target['scaffold_smiles'].nunique()

                        # Find scaffolds unique to active/inactive
                        active_scaffold_set = set(active_scaffolds.unique())
                        inactive_scaffold_set = set(inactive_scaffolds.unique())

                        active_only = len(active_scaffold_set - inactive_scaffold_set)
                        inactive_only = len(inactive_scaffold_set - active_scaffold_set)
                        shared = len(active_scaffold_set & inactive_scaffold_set)

                        scaffold_target_analysis[target] = {
                            'active_compounds': active_mask.sum(),
                            'inactive_compounds': inactive_mask.sum(),
                            'active_scaffolds': active_unique,
                            'inactive_scaffolds': inactive_unique,
                            'total_scaffolds': total_unique,
                            'active_only_scaffolds': active_only,
                            'inactive_only_scaffolds': inactive_only,
                            'shared_scaffolds': shared
                        }

        # Print scaffold-target analysis
        print("\n📊 Scaffold-Target Activity Analysis:")
        print("=" * 100)
        print(f"{'Target':<20} {'Active':<8} {'Inactive':<8} {'Active_Scaf':<12} {'Inactive_Scaf':<14} {'Shared':<8}")
        print("-" * 100)

        for target, analysis in scaffold_target_analysis.items():
            print(f"{target:<20} {analysis['active_compounds']:<8} {analysis['inactive_compounds']:<8} "
                  f"{analysis['active_scaffolds']:<12} {analysis['inactive_scaffolds']:<14} {analysis['shared_scaffolds']:<8}")

        self.scaffold_target_analysis = scaffold_target_analysis

    def _analyze_molecular_properties(self):
        """Analyze molecular properties and Lipinski's Rule of Five per target"""
        print("\n🧪 Molecular Properties Analysis per Target:")
        print("-" * 50)

        compounds = self.data['compounds']
        targets = self.data['targets']

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
        self.molecular_props_df.set_index('compound_id', inplace=True)

        # Target-specific molecular properties analysis
        print("\n📊 Target-Specific Molecular Properties Analysis:")
        print("=" * 100)

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                valid_indices = target_data.index.intersection(self.molecular_props_df.index)
                if len(valid_indices) > 0:
                    target_values = target_data.loc[valid_indices]
                    props_with_target = self.molecular_props_df.loc[valid_indices]

                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        print(f"\n🎯 {target}:")
                        print(f"Active compounds: {active_mask.sum()}, Inactive compounds: {inactive_mask.sum()}")
                        print("-" * 60)
                        print(f"{'Property':<15} {'Active_Mean':<12} {'Inactive_Mean':<14} {'Difference':<12} {'%_Diff':<8}")
                        print("-" * 60)

                        for prop in ['molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'aromatic_rings']:
                            active_values = props_with_target.loc[active_mask, prop]
                            inactive_values = props_with_target.loc[inactive_mask, prop]

                            active_mean = active_values.mean()
                            inactive_mean = inactive_values.mean()
                            difference = active_mean - inactive_mean
                            pct_diff = (difference / inactive_mean * 100) if inactive_mean != 0 else 0

                            print(f"{prop:<15} {active_mean:<12.2f} {inactive_mean:<14.2f} {difference:<12.2f} {pct_diff:<8.1f}%")

        # Target-specific Lipinski analysis
        self._analyze_lipinski_rules_per_target()

    def _analyze_lipinski_rules_per_target(self):
        """Analyze Lipinski's Rule of Five compliance per target"""
        print("\n📋 Lipinski's Rule of Five Analysis per Target:")
        print("-" * 60)

        targets = self.data['targets']
        df = self.molecular_props_df

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                valid_indices = target_data.index.intersection(df.index)
                if len(valid_indices) > 0:
                    target_values = target_data.loc[valid_indices]
                    props_with_target = df.loc[valid_indices]

                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        print(f"\n🎯 {target}:")
                        print(f"Active: {active_mask.sum()}, Inactive: {inactive_mask.sum()}")
                        print("-" * 50)

                        # Analyze each rule for active vs inactive
                        rules = [
                            ('MW ≤ 500', 'molecular_weight', lambda x: x <= 500),
                            ('LogP ≤ 5', 'logp', lambda x: x <= 5),
                            ('HBD ≤ 5', 'hbd', lambda x: x <= 5),
                            ('HBA ≤ 10', 'hba', lambda x: x <= 10)
                        ]

                        print(f"{'Rule':<12} {'Active_Compliant':<16} {'Inactive_Compliant':<18} {'Active_%':<10} {'Inactive_%':<12}")
                        print("-" * 70)

                        for rule_name, prop, rule_func in rules:
                            active_compliant = rule_func(props_with_target.loc[active_mask, prop]).sum()
                            inactive_compliant = rule_func(props_with_target.loc[inactive_mask, prop]).sum()

                            active_pct = (active_compliant / active_mask.sum()) * 100
                            inactive_pct = (inactive_compliant / inactive_mask.sum()) * 100

                            print(f"{rule_name:<12} {active_compliant:<16} {inactive_compliant:<18} {active_pct:<10.1f} {inactive_pct:<12.1f}")

                        # Overall compliance
                        active_compliance = 0
                        inactive_compliance = 0

                        for _, prop, rule_func in rules:
                            active_compliance += rule_func(props_with_target.loc[active_mask, prop])
                            inactive_compliance += rule_func(props_with_target.loc[inactive_mask, prop])

                        active_all_rules = (active_compliance == 4).sum()
                        inactive_all_rules = (inactive_compliance == 4).sum()

                        print(f"\nAll 4 Rules - Active: {active_all_rules} ({active_all_rules/active_mask.sum()*100:.1f}%)")
                        print(f"All 4 Rules - Inactive: {inactive_all_rules} ({inactive_all_rules/inactive_mask.sum()*100:.1f}%)")

    def _generate_fingerprints(self):
        """Generate molecular fingerprints for diversity analysis"""
        print("\n🔬 Generating Molecular Fingerprints...")

        compounds = self.data['compounds']
        fingerprints = []
        valid_mols = []

        # Generate Morgan fingerprints (ECFP-like)
        for mol in compounds['molecule']:
            if mol is not None:
                # Generate Morgan fingerprint with radius 2, 2048 bits
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
                valid_mols.append(mol)

        self.fingerprints = fingerprints
        print(f"✅ Generated {len(fingerprints):,} Morgan fingerprints")

        # Convert fingerprints to numpy array for analysis
        fp_array = np.array([list(fp) for fp in fingerprints])
        self.fp_array = fp_array

        print(f"📊 Fingerprint array shape: {fp_array.shape}")

    def _analyze_molecular_diversity(self):
        """Analyze molecular diversity and similarity per target"""
        print("\n🌐 Molecular Diversity Analysis per Target:")
        print("-" * 50)

        if self.fp_array is None:
            print("⚠️ Fingerprints not generated. Running fingerprint generation first...")
            self._generate_fingerprints()

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Overall diversity analysis
        print(f"\n📊 Overall Molecular Diversity:")
        print(f"Fingerprint array shape: {self.fp_array.shape}")
        print(f"Average bit density: {np.mean(self.fp_array):.4f}")
        print(f"Active bits: {np.sum(np.sum(self.fp_array, axis=0) > 0)}")

        # Target-specific diversity analysis
        print(f"\n🎯 Target-Specific Diversity Analysis:")
        print("=" * 100)

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                valid_indices = target_data.index.intersection(compounds.index)
                if len(valid_indices) > 0:
                    target_values = target_data.loc[valid_indices]
                    fp_indices = [i for i, idx in enumerate(compounds.index) if idx in valid_indices]

                    if len(fp_indices) > 0:
                        target_fps = self.fp_array[fp_indices]
                        active_mask = target_values == 1
                        inactive_mask = target_values == 0

                        if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                            active_fps = target_fps[active_mask.values]
                            inactive_fps = target_fps[inactive_mask.values]

                            print(f"\n🎯 {target}:")
                            print(f"Active compounds: {active_mask.sum()}, Inactive compounds: {inactive_mask.sum()}")
                            print("-" * 60)

                            # Bit density analysis
                            active_bit_density = np.mean(active_fps)
                            inactive_bit_density = np.mean(inactive_fps)
                            active_active_bits = np.sum(np.sum(active_fps, axis=0) > 0)
                            inactive_active_bits = np.sum(np.sum(inactive_fps, axis=0) > 0)

                            print(f"Bit Density - Active: {active_bit_density:.4f}, Inactive: {inactive_bit_density:.4f}")
                            print(f"Active Bits - Active compounds: {active_active_bits}, Inactive compounds: {inactive_active_bits}")

                            # Similarity analysis within groups
                            if len(active_fps) > 1:
                                active_similarities = self._calculate_pairwise_similarities(active_fps)
                                print(f"Active-Active Similarity - Mean: {np.mean(active_similarities):.3f}, "
                                      f"Max: {np.max(active_similarities):.3f}")

                            if len(inactive_fps) > 1:
                                inactive_similarities = self._calculate_pairwise_similarities(inactive_fps)
                                print(f"Inactive-Inactive Similarity - Mean: {np.mean(inactive_similarities):.3f}, "
                                      f"Max: {np.max(inactive_similarities):.3f}")

                            # Cross-group similarity
                            if len(active_fps) > 0 and len(inactive_fps) > 0:
                                cross_similarities = self._calculate_cross_similarities(active_fps, inactive_fps)
                                print(f"Active-Inactive Similarity - Mean: {np.mean(cross_similarities):.3f}, "
                                      f"Max: {np.max(cross_similarities):.3f}")

        print("✅ Target-specific diversity analysis completed!")

    def _calculate_pairwise_similarities(self, fps):
        """Calculate pairwise Tanimoto similarities for a set of fingerprints"""
        similarities = []
        n = len(fps)

        for i in range(n):
            for j in range(i+1, min(i+100, n)):  # Limit to avoid memory issues
                # Calculate Tanimoto similarity using numpy operations
                fp1 = fps[i].astype(bool)
                fp2 = fps[j].astype(bool)

                intersection = np.sum(fp1 & fp2)
                union = np.sum(fp1 | fp2)

                if union > 0:
                    sim = intersection / union
                    similarities.append(sim)

        return similarities

    def _calculate_cross_similarities(self, fps1, fps2):
        """Calculate cross-group similarities between two fingerprint sets"""
        similarities = []

        for fp1 in fps1[:100]:  # Limit to avoid memory issues
            for fp2 in fps2[:100]:
                # Calculate Tanimoto similarity using numpy operations
                fp1_bool = fp1.astype(bool)
                fp2_bool = fp2.astype(bool)

                intersection = np.sum(fp1_bool & fp2_bool)
                union = np.sum(fp1_bool | fp2_bool)

                if union > 0:
                    sim = intersection / union
                    similarities.append(sim)

        return similarities

    def _perform_clustering_analysis(self):
        """Perform clustering analysis with target-specific classifications"""
        print("\n🔍 Performing Target-Specific Clustering Analysis...")

        if self.fp_array is None:
            print("⚠️ Fingerprints not generated. Running fingerprint generation first...")
            self._generate_fingerprints()

        # Perform PCA for dimensionality reduction
        print("📊 Performing PCA dimensionality reduction...")
        pca = PCA(n_components=2)
        fp_pca = pca.fit_transform(self.fp_array)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

        # Target-specific clustering analysis
        compounds = self.data['compounds']
        targets = self.data['targets']

        print(f"\n🎯 Target-Specific Clustering Analysis:")
        print("=" * 100)

        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                valid_indices = target_data.index.intersection(compounds.index)
                if len(valid_indices) > 0:
                    target_values = target_data.loc[valid_indices]
                    fp_indices = [i for i, idx in enumerate(compounds.index) if idx in valid_indices]

                    if len(fp_indices) > 0:
                        target_fps = self.fp_array[fp_indices]
                        active_mask = target_values == 1
                        inactive_mask = target_values == 0

                        if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                            print(f"\n🎯 {target}:")
                            print(f"Active compounds: {active_mask.sum()}, Inactive compounds: {inactive_mask.sum()}")

                            # Separate clustering for active and inactive
                            active_fps = target_fps[active_mask.values]
                            inactive_fps = target_fps[inactive_mask.values]

                            # Clustering for active compounds
                            if len(active_fps) > 10:
                                print(f"\nActive Compounds Clustering:")
                                active_pca = PCA(n_components=2).fit_transform(active_fps)
                                active_clusters = self._find_optimal_clusters(active_pca, max_clusters=min(10, len(active_fps)//10))
                                print(f"Optimal clusters for active compounds: {active_clusters}")

                            # Clustering for inactive compounds
                            if len(inactive_fps) > 10:
                                print(f"\nInactive Compounds Clustering:")
                                inactive_pca = PCA(n_components=2).fit_transform(inactive_fps)
                                inactive_clusters = self._find_optimal_clusters(inactive_pca, max_clusters=min(10, len(inactive_fps)//10))
                                print(f"Optimal clusters for inactive compounds: {inactive_clusters}")

        # Overall clustering analysis
        print(f"\n📊 Overall Clustering Analysis:")
        print("🔍 Performing K-means clustering...")

        # Find optimal number of clusters using silhouette analysis
        max_clusters = min(25, len(fp_pca) // 100)
        silhouette_scores = []

        for n_clusters in range(5, max_clusters + 1, 5):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(fp_pca)
            silhouette_avg = silhouette_score(fp_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"  {n_clusters} clusters: Silhouette score = {silhouette_avg:.3f}")

        optimal_clusters = 5 + (np.argmax(silhouette_scores) * 5)
        print(f"✅ Optimal number of clusters: {optimal_clusters}")

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(fp_pca)

        # Analyze cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\n📊 Cluster Size Analysis:")
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {label}: {count:,} molecules ({percentage:.1f}%)")

        # Target distribution in clusters
        self._analyze_target_distribution_in_clusters(cluster_labels)

        # Store clustering results
        self.cluster_results = {
            'labels': cluster_labels,
            'pca_coords': fp_pca,
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores
        }

    def _find_optimal_clusters(self, data, max_clusters=10):
        """Find optimal number of clusters using silhouette analysis"""
        if len(data) < 10:
            return 1

        silhouette_scores = []
        for n_clusters in range(2, min(max_clusters + 1, len(data) // 10 + 1)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        if silhouette_scores:
            optimal = 2 + np.argmax(silhouette_scores)
            return optimal
        return 1

    def _analyze_target_distribution_in_clusters(self, cluster_labels):
        """Analyze how targets are distributed across clusters"""
        print("\n🎯 Target Distribution in Clusters:")
        print("-" * 40)

        if self.cluster_results is None:
            return

        labels = cluster_labels
        targets = self.data['targets']

        # Create cluster-target analysis
        cluster_target_data = []

        for cluster_id in range(self.cluster_results['optimal_clusters']):
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            cluster_targets = targets[cluster_mask]

            for target in self.data_loader.target_columns:
                target_data = cluster_targets[target].dropna()
                if len(target_data) > 0:
                    positive_ratio = target_data.mean()
                    cluster_target_data.append({
                        'cluster': cluster_id,
                        'target': target,
                        'cluster_size': cluster_size,
                        'positive_ratio': positive_ratio,
                        'positive_count': int(target_data.sum()),
                        'total_count': len(target_data)
                    })

        self.cluster_target_df = pd.DataFrame(cluster_target_data)

        # Find clusters with highest activity for each target
        print("Top clusters by target activity:")
        for target in self.data_loader.target_columns:
            target_data = self.cluster_target_df[self.cluster_target_df['target'] == target]
            if len(target_data) > 0:
                top_cluster = target_data.loc[target_data['positive_ratio'].idxmax()]
                print(f"  {target}: Cluster {top_cluster['cluster']} "
                      f"({top_cluster['positive_ratio']*100:.1f}% positive, "
                      f"{top_cluster['positive_count']}/{top_cluster['total_count']} compounds)")

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating Enhanced Visualizations...")

        # Set up the plotting style
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.2)
        sns.set_palette('colorblind')
        plt.rcParams.update({
            'axes.titlesize': 18,
            'axes.titleweight': 'bold',
            'axes.labelsize': 16,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.5,
        })

        # Create a comprehensive dashboard
        self._create_molecular_properties_plots()
        self._create_target_distribution_plots()
        self._create_lipinski_analysis_plots()
        self._create_correlation_heatmap()
        self._create_target_specific_analysis()
        self._create_fingerprint_analysis_plots()
        self._create_clustering_visualizations()
        self._create_active_inactive_visualizations()
        self._create_scaffold_visualizations()

        print("✅ Enhanced Visualizations created successfully!")

    def _create_molecular_properties_plots(self):
        """Create target-specific molecular properties plots with active/inactive coloring"""
        print("\n📊 Creating Target-Specific Molecular Properties Plots...")

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Get all targets with activity data
        target_activity = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                target_activity[target] = target_data.mean()

        all_targets = sorted(target_activity.items(), key=lambda x: x[1], reverse=True)

        # Create comprehensive molecular properties plots
        properties = ['molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'aromatic_rings']
        titles = ['Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 'TPSA', 'Rotatable Bonds', 'Aromatic Rings']

        # Create subplots for each target
        for target, activity in all_targets[:6]:  # Top 6 targets
            target_data = targets[target].dropna()
            valid_indices = target_data.index.intersection(compounds.index)

            if len(valid_indices) > 0:
                target_values = target_data.loc[valid_indices]
                # Ensure we only use indices that exist in molecular_props_df
                valid_prop_indices = valid_indices.intersection(self.molecular_props_df.index)

                if len(valid_prop_indices) > 0:
                    target_values = target_values.loc[valid_prop_indices]
                    props_with_target = self.molecular_props_df.loc[valid_prop_indices]

                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        # Create subplot for this target
                        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                        fig.suptitle(f'Molecular Properties Analysis - {target}', fontsize=16, fontweight='bold')

                        for i, (prop, title) in enumerate(zip(properties, titles)):
                            row, col = i // 4, i % 4
                            ax = axes[row, col]

                            # Plot active and inactive distributions
                            active_values = props_with_target.loc[active_mask, prop]
                            inactive_values = props_with_target.loc[inactive_mask, prop]

                            # Create step histograms for better visibility
                            ax.hist(active_values, bins=20, alpha=0.7, color='red',
                                   label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                            ax.hist(inactive_values, bins=20, alpha=0.7, color='blue',
                                   label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                                   linestyle='--', linewidth=1)

                            # Add statistics
                            active_mean = active_values.mean()
                            inactive_mean = inactive_values.mean()
                            ax.axvline(active_mean, color='red', linestyle='-', alpha=0.8, linewidth=1)
                            ax.axvline(inactive_mean, color='blue', linestyle='--', alpha=0.8, linewidth=1)

                            ax.set_title(f'{title}\nActive={active_mean:.1f}, Inactive={inactive_mean:.1f}',
                                       fontweight='bold', fontsize=10)
                            ax.set_xlabel(prop.replace('_', ' ').title())
                            ax.set_ylabel('Density')
                            ax.legend(fontsize=8)

                        # Remove empty subplot
                        axes[1, 3].remove()

                        plt.tight_layout()
                        plt.savefig(f'results/molecular_properties_{target.lower().replace("-", "_")}.png',
                                  dpi=600, bbox_inches='tight')
                        plt.savefig(f'results/molecular_properties_{target.lower().replace("-", "_")}.svg',
                                  bbox_inches='tight')
                        plt.close()

        # Also create the original version without target coloring
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Overall Molecular Properties Distribution', fontsize=16, fontweight='bold')

        for i, (prop, title) in enumerate(zip(properties, titles)):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            sns.histplot(self.molecular_props_df[prop], kde=True, ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(prop.replace('_', ' ').title())
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('results/molecular_properties_distribution.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/molecular_properties_distribution.svg', bbox_inches='tight')
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
        plt.savefig('results/target_distribution.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/target_distribution.svg', bbox_inches='tight')
        plt.close()
        # Figure caption: Positive ratio for each Tox21 target endpoint

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
        plt.savefig('results/lipinski_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/lipinski_analysis.svg', bbox_inches='tight')
        plt.close()
        # Figure caption: Lipinski's Rule of Five compliance for the dataset

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
        plt.savefig('results/molecular_properties_correlation.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/molecular_properties_correlation.svg', bbox_inches='tight')
        plt.close()
        # Figure caption: Correlation matrix of molecular properties

    def _create_target_specific_analysis(self):
        """Create target-specific molecular properties analysis"""
        print("\n🎯 Creating Target-Specific Analysis...")

        # Merge molecular properties with target data
        compounds = self.data['compounds']
        targets = self.data['targets']

        # Add molecular properties to compounds dataframe
        compounds_with_props = compounds.copy()
        for col in self.molecular_props_df.columns:
            if col != 'compound_id':
                compounds_with_props[col] = self.molecular_props_df[col]

        # Create target-specific analysis for each endpoint
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Molecular Properties by Target Endpoint', fontsize=16, fontweight='bold')

        for i, target in enumerate(self.data_loader.target_columns):
            row, col = i // 4, i % 4
            ax = axes[row, col]

            # Get data for this target
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                # Merge with molecular properties
                merged_data = pd.concat([compounds_with_props, target_data], axis=1)
                merged_data = merged_data.dropna(subset=[target])

                # Create box plot for molecular weight by target value
                positive_data = merged_data[merged_data[target] == 1]['molecular_weight']
                negative_data = merged_data[merged_data[target] == 0]['molecular_weight']

                if len(positive_data) > 0 and len(negative_data) > 0:
                    data_to_plot = [negative_data, positive_data]
                    labels = ['Negative', 'Positive']
                    ax.boxplot(data_to_plot, labels=labels)
                    ax.set_title(f'{target}\nMW Distribution')
                    ax.set_ylabel('Molecular Weight')

        plt.tight_layout()
        plt.savefig('results/target_specific_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/target_specific_analysis.svg', bbox_inches='tight')
        plt.close()
        # Figure caption: Boxplots of molecular weight by target and activity

    def _create_fingerprint_analysis_plots(self):
        """Create target-specific fingerprint analysis plots with active/inactive coloring"""
        if self.fp_array is None:
            return

        print("\n🔬 Creating Target-Specific Fingerprint Analysis Plots...")

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Get all targets with activity data
        target_activity = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                target_activity[target] = target_data.mean()

        all_targets = sorted(target_activity.items(), key=lambda x: x[1], reverse=True)

        # Create target-specific fingerprint analysis
        for target, activity in all_targets[:6]:  # Top 6 targets
            target_data = targets[target].dropna()
            valid_indices = target_data.index.intersection(compounds.index)

            if len(valid_indices) > 0:
                target_values = target_data.loc[valid_indices]
                fp_indices = [i for i, idx in enumerate(compounds.index) if idx in valid_indices]

                if len(fp_indices) > 0:
                    target_fps = self.fp_array[fp_indices]
                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        active_fps = target_fps[active_mask.values]
                        inactive_fps = target_fps[inactive_mask.values]

                        # Create comprehensive fingerprint analysis for this target
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        fig.suptitle(f'Fingerprint Analysis - {target}', fontsize=16, fontweight='bold')

                        # 1. Bit density comparison
                        ax1 = axes[0, 0]
                        active_bit_density = np.mean(active_fps, axis=0)
                        inactive_bit_density = np.mean(inactive_fps, axis=0)

                        ax1.hist(active_bit_density, bins=50, alpha=0.7, color='red',
                               label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                        ax1.hist(inactive_bit_density, bins=50, alpha=0.7, color='blue',
                               label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                               linestyle='--', linewidth=1)
                        ax1.set_title('Bit Density Distribution')
                        ax1.set_xlabel('Bit Density')
                        ax1.set_ylabel('Density')
                        ax1.legend()

                        # 2. Active bits comparison
                        ax2 = axes[0, 1]
                        active_bits = np.sum(active_fps, axis=0)
                        inactive_bits = np.sum(inactive_fps, axis=0)

                        ax2.hist(active_bits[active_bits > 0], bins=50, alpha=0.7, color='red',
                               label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                        ax2.hist(inactive_bits[inactive_bits > 0], bins=50, alpha=0.7, color='blue',
                               label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                               linestyle='--', linewidth=1)
                        ax2.set_title('Active Bits Distribution')
                        ax2.set_xlabel('Active Bits Count')
                        ax2.set_ylabel('Density')
                        ax2.legend()

                        # 3. Fingerprint complexity
                        ax3 = axes[0, 2]
                        active_complexity = np.sum(active_fps, axis=1)
                        inactive_complexity = np.sum(inactive_fps, axis=1)

                        ax3.hist(active_complexity, bins=30, alpha=0.7, color='red',
                               label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                        ax3.hist(inactive_complexity, bins=30, alpha=0.7, color='blue',
                               label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                               linestyle='--', linewidth=1)
                        ax3.set_title('Fingerprint Complexity')
                        ax3.set_xlabel('Number of Set Bits')
                        ax3.set_ylabel('Density')
                        ax3.legend()

                        # 4. PCA of fingerprints colored by activity
                        ax4 = axes[1, 0]
                        pca = PCA(n_components=2)
                        combined_fps = np.vstack([active_fps, inactive_fps])
                        fp_pca = pca.fit_transform(combined_fps)

                        # Plot active compounds
                        ax4.scatter(fp_pca[:len(active_fps), 0], fp_pca[:len(active_fps), 1],
                                  c='red', alpha=0.7, s=20, label=f'Active ({active_mask.sum()})')
                        # Plot inactive compounds
                        ax4.scatter(fp_pca[len(active_fps):, 0], fp_pca[len(active_fps):, 1],
                                  c='blue', alpha=0.7, s=20, label=f'Inactive ({inactive_mask.sum()})')
                        ax4.set_title('PCA of Fingerprints')
                        ax4.set_xlabel('PC1')
                        ax4.set_ylabel('PC2')
                        ax4.legend()

                        # 5. Molecular weight vs fingerprint complexity
                        ax5 = axes[1, 1]
                        active_mw = self.molecular_props_df.loc[valid_indices[active_mask], 'molecular_weight']
                        inactive_mw = self.molecular_props_df.loc[valid_indices[inactive_mask], 'molecular_weight']

                        ax5.scatter(active_mw, active_complexity, c='red', alpha=0.7, s=20,
                                  label=f'Active ({active_mask.sum()})')
                        ax5.scatter(inactive_mw, inactive_complexity, c='blue', alpha=0.7, s=20,
                                  label=f'Inactive ({inactive_mask.sum()})')
                        ax5.set_title('MW vs Fingerprint Complexity')
                        ax5.set_xlabel('Molecular Weight')
                        ax5.set_ylabel('Fingerprint Complexity')
                        ax5.legend()

                        # 6. Similarity distribution
                        ax6 = axes[1, 2]
                        if len(active_fps) > 1:
                            active_similarities = self._calculate_pairwise_similarities(active_fps)
                            ax6.hist(active_similarities, bins=30, alpha=0.7, color='red',
                                   label=f'Active-Active ({len(active_similarities)})', density=True)

                        if len(inactive_fps) > 1:
                            inactive_similarities = self._calculate_pairwise_similarities(inactive_fps)
                            ax6.hist(inactive_similarities, bins=30, alpha=0.7, color='blue',
                                   label=f'Inactive-Inactive ({len(inactive_similarities)})', density=True)

                        ax6.set_title('Similarity Distribution')
                        ax6.set_xlabel('Tanimoto Similarity')
                        ax6.set_ylabel('Density')
                        ax6.legend()

                        plt.tight_layout()
                        plt.savefig(f'results/fingerprint_analysis_{target.lower().replace("-", "_")}.png',
                                  dpi=600, bbox_inches='tight')
                        plt.savefig(f'results/fingerprint_analysis_{target.lower().replace("-", "_")}.svg',
                                  bbox_inches='tight')
                        plt.close()

    def _create_clustering_visualizations(self):
        """Create target-specific clustering visualizations with active/inactive subplotting"""
        if self.cluster_results is None:
            return

        print("\n🔍 Creating Target-Specific Clustering Visualizations...")

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Get all targets with activity data
        target_activity = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                target_activity[target] = target_data.mean()

        all_targets = sorted(target_activity.items(), key=lambda x: x[1], reverse=True)

        # Create target-specific clustering analysis
        for target, activity in all_targets[:6]:  # Top 6 targets
            target_data = targets[target].dropna()
            valid_indices = target_data.index.intersection(compounds.index)

            if len(valid_indices) > 0:
                target_values = target_data.loc[valid_indices]
                cluster_indices = [i for i, idx in enumerate(compounds.index) if idx in valid_indices]

                if len(cluster_indices) > 0:
                    target_clusters = self.cluster_results['labels'][cluster_indices]
                    target_pca = self.cluster_results['pca_coords'][cluster_indices]
                    active_mask = target_values == 1
                    inactive_mask = target_values == 0

                    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                        # Create comprehensive clustering analysis for this target
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        fig.suptitle(f'Clustering Analysis - {target}', fontsize=16, fontweight='bold')

                        # 1. Overall clustering with target activity
                        ax1 = axes[0, 0]
                        active_indices = np.where(active_mask.values)[0]
                        inactive_indices = np.where(inactive_mask.values)[0]

                        # Plot active compounds
                        ax1.scatter(target_pca[active_indices, 0], target_pca[active_indices, 1],
                                  c='red', alpha=0.7, s=30, label=f'Active ({active_mask.sum()})')
                        # Plot inactive compounds
                        ax1.scatter(target_pca[inactive_indices, 0], target_pca[inactive_indices, 1],
                                  c='blue', alpha=0.7, s=30, label=f'Inactive ({inactive_mask.sum()})')
                        ax1.set_title('PCA Clustering with Target Activity')
                        ax1.set_xlabel('PC1')
                        ax1.set_ylabel('PC2')
                        ax1.legend()

                        # 2. Cluster distribution by activity
                        ax2 = axes[0, 1]
                        active_clusters = target_clusters[active_indices]
                        inactive_clusters = target_clusters[inactive_indices]

                        unique_clusters = np.unique(target_clusters)
                        active_counts = [np.sum(active_clusters == c) for c in unique_clusters]
                        inactive_counts = [np.sum(inactive_clusters == c) for c in unique_clusters]

                        x = np.arange(len(unique_clusters))
                        width = 0.35

                        ax2.bar(x - width/2, active_counts, width, label='Active', color='red', alpha=0.7)
                        ax2.bar(x + width/2, inactive_counts, width, label='Inactive', color='blue', alpha=0.7)
                        ax2.set_title('Cluster Distribution by Activity')
                        ax2.set_xlabel('Cluster')
                        ax2.set_ylabel('Number of Compounds')
                        ax2.set_xticks(x)
                        ax2.set_xticklabels([f'C{c}' for c in unique_clusters])
                        ax2.legend()

                        # 3. Activity ratio by cluster
                        ax3 = axes[0, 2]
                        activity_ratios = []
                        for c in unique_clusters:
                            cluster_mask = target_clusters == c
                            cluster_active = np.sum(active_mask.values & cluster_mask)
                            cluster_total = np.sum(cluster_mask)
                            ratio = cluster_active / cluster_total if cluster_total > 0 else 0
                            activity_ratios.append(ratio)

                        colors = ['red' if ratio > 0.5 else 'blue' for ratio in activity_ratios]
                        bars = ax3.bar(unique_clusters, activity_ratios, color=colors, alpha=0.7)
                        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
                        ax3.set_title('Activity Ratio by Cluster')
                        ax3.set_xlabel('Cluster')
                        ax3.set_ylabel('Active Ratio')
                        ax3.legend()

                        # 4. Active compounds clustering
                        ax4 = axes[1, 0]
                        if len(active_indices) > 0:
                            active_pca = target_pca[active_indices]
                            active_cluster_labels = target_clusters[active_indices]

                            for cluster_id in np.unique(active_cluster_labels):
                                cluster_mask = active_cluster_labels == cluster_id
                                ax4.scatter(active_pca[cluster_mask, 0], active_pca[cluster_mask, 1],
                                          alpha=0.7, s=30, label=f'Cluster {cluster_id}')

                            ax4.set_title('Active Compounds Clustering')
                            ax4.set_xlabel('PC1')
                            ax4.set_ylabel('PC2')
                            ax4.legend()

                        # 5. Inactive compounds clustering
                        ax5 = axes[1, 1]
                        if len(inactive_indices) > 0:
                            inactive_pca = target_pca[inactive_indices]
                            inactive_cluster_labels = target_clusters[inactive_indices]

                            for cluster_id in np.unique(inactive_cluster_labels):
                                cluster_mask = inactive_cluster_labels == cluster_id
                                ax5.scatter(inactive_pca[cluster_mask, 0], inactive_pca[cluster_mask, 1],
                                          alpha=0.7, s=30, label=f'Cluster {cluster_id}')

                            ax5.set_title('Inactive Compounds Clustering')
                            ax5.set_xlabel('PC1')
                            ax5.set_ylabel('PC2')
                            ax5.legend()

                        # 6. Molecular weight vs cluster by activity
                        ax6 = axes[1, 2]
                        active_mw = self.molecular_props_df.loc[valid_indices[active_mask], 'molecular_weight']
                        inactive_mw = self.molecular_props_df.loc[valid_indices[inactive_mask], 'molecular_weight']

                        ax6.scatter(active_clusters, active_mw, c='red', alpha=0.7, s=20,
                                  label=f'Active ({active_mask.sum()})')
                        ax6.scatter(inactive_clusters, inactive_mw, c='blue', alpha=0.7, s=20,
                                  label=f'Inactive ({inactive_mask.sum()})')
                        ax6.set_title('MW vs Cluster by Activity')
                        ax6.set_xlabel('Cluster')
                        ax6.set_ylabel('Molecular Weight')
                        ax6.legend()

                        plt.tight_layout()
                        plt.savefig(f'results/clustering_analysis_{target.lower().replace("-", "_")}.png',
                                  dpi=600, bbox_inches='tight')
                        plt.savefig(f'results/clustering_analysis_{target.lower().replace("-", "_")}.svg',
                                  bbox_inches='tight')
                        plt.close()

        # Also create overall clustering visualization
        self._create_overall_clustering_visualization()

    def _create_overall_clustering_visualization(self):
        """Create overall clustering visualization"""
        if self.cluster_results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Clustering Analysis', fontsize=16, fontweight='bold')

        pca_coords = self.cluster_results['pca_coords']
        labels = self.cluster_results['labels']

        # 1. Overall clustering
        ax1 = axes[0, 0]
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax1.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       c=[color], alpha=0.7, s=20, label=f'Cluster {label}')

        ax1.set_title('Overall Clustering')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Cluster sizes
        ax2 = axes[0, 1]
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        ax2.bar(unique_labels, cluster_sizes, alpha=0.7)
        ax2.set_title('Cluster Sizes')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Compounds')

        # 3. Target distribution in clusters
        ax3 = axes[1, 0]
        targets = self.data['targets']

        # Show distribution for top target
        top_target = self.data_loader.target_columns[0]
        target_data = targets[top_target].dropna()
        valid_indices = target_data.index.intersection(self.data['compounds'].index)

        if len(valid_indices) > 0:
            target_values = target_data.loc[valid_indices]
            cluster_indices = [i for i, idx in enumerate(self.data['compounds'].index) if idx in valid_indices]

            if len(cluster_indices) > 0:
                target_clusters = labels[cluster_indices]
                active_mask = target_values == 1

                for cluster_id in unique_labels:
                    cluster_mask = target_clusters == cluster_id
                    active_ratio = np.sum(active_mask.values & cluster_mask) / np.sum(cluster_mask) if np.sum(cluster_mask) > 0 else 0
                    ax3.bar(cluster_id, active_ratio, alpha=0.7, color='red' if active_ratio > 0.5 else 'blue')

                ax3.set_title(f'Activity Ratio by Cluster ({top_target})')
                ax3.set_xlabel('Cluster')
                ax3.set_ylabel('Active Ratio')
                ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

        # 4. Silhouette scores
        ax4 = axes[1, 1]
        if hasattr(self.cluster_results, 'silhouette_scores'):
            n_clusters_range = range(5, len(self.cluster_results['silhouette_scores']) * 5 + 1, 5)
            ax4.plot(n_clusters_range, self.cluster_results['silhouette_scores'], 'bo-')
            ax4.set_title('Silhouette Score vs Number of Clusters')
            ax4.set_xlabel('Number of Clusters')
            ax4.set_ylabel('Silhouette Score')

        plt.tight_layout()
        plt.savefig('results/overall_clustering_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/overall_clustering_analysis.svg', bbox_inches='tight')
        plt.close()

    def _create_active_inactive_visualizations(self):
        """Create comprehensive active/inactive analysis visualizations"""
        print("\n🎯 Creating Active/Inactive Analysis Visualizations...")

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Create comprehensive active/inactive analysis
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Comprehensive Active/Inactive Analysis by Target', fontsize=16, fontweight='bold')

        # Get all targets with activity data
        target_activity = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                target_activity[target] = target_data.mean()

        all_targets = sorted(target_activity.items(), key=lambda x: x[1], reverse=True)

        # Create color palette
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values()) + ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i, (target, activity) in enumerate(all_targets[:12]):  # Top 12 targets
            row, col = i // 4, i % 4
            ax = axes[row, col]

            target_data = targets[target].dropna()
            valid_indices = target_data.index.intersection(compounds.index)

            if len(valid_indices) > 0:
                target_values = target_data.loc[valid_indices]
                props_with_target = self.molecular_props_df.loc[valid_indices]

                active_mask = target_values == 1
                inactive_mask = target_values == 0

                if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                    # Plot molecular weight distribution for active vs inactive
                    active_mw = props_with_target.loc[active_mask, 'molecular_weight']
                    inactive_mw = props_with_target.loc[inactive_mask, 'molecular_weight']

                    color = colors[i % len(colors)]

                    # Plot histograms
                    ax.hist(active_mw, bins=20, alpha=0.7, color=color,
                           label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                    ax.hist(inactive_mw, bins=20, alpha=0.7, color=color,
                           label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                           linestyle='--', linewidth=1)

                    # Add statistics
                    active_mean = active_mw.mean()
                    inactive_mean = inactive_mw.mean()
                    ax.axvline(active_mean, color=color, linestyle='-', alpha=0.8, linewidth=1)
                    ax.axvline(inactive_mean, color=color, linestyle='--', alpha=0.8, linewidth=1)

                    ax.set_title(f'{target}\nMW: Active={active_mean:.0f}, Inactive={inactive_mean:.0f}',
                               fontweight='bold', fontsize=10)
                    ax.set_xlabel('Molecular Weight')
                    ax.set_ylabel('Density')
                    ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig('results/active_inactive_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/active_inactive_analysis.svg', bbox_inches='tight')
        plt.close()

        # Create SAR heatmap
        if hasattr(self, 'scaffold_target_analysis'):
            self._create_scaffold_target_activity_heatmap()
        else:
            # Run scaffold-target activity analysis if not already done
            self._analyze_scaffold_target_activity()
            self._create_scaffold_target_activity_heatmap()

        # Create class imbalance visualization
        self._create_class_imbalance_visualization()

        print("✅ Active/Inactive Analysis Visualizations created successfully!")

    def _create_scaffold_target_activity_heatmap(self):
        """Create scaffold-target activity heatmap"""
        if not hasattr(self, 'scaffold_target_analysis'):
            return

        # Prepare data for heatmap
        targets = list(self.scaffold_target_analysis.keys())

        # Create scaffold diversity matrix
        diversity_matrix = np.zeros((len(targets), 1))

        for i, target in enumerate(targets):
            analysis = self.scaffold_target_analysis[target]
            total_scaffolds = analysis['active_scaffolds'] + analysis['inactive_scaffolds']
            if total_scaffolds > 0:
                diversity_ratio = analysis['active_scaffolds'] / total_scaffolds
                diversity_matrix[i, 0] = diversity_ratio

        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Scaffold-Target Activity Analysis', fontsize=16, fontweight='bold')

        # Scaffold diversity heatmap
        im = ax.imshow(diversity_matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title('Active Scaffold Ratio by Target', fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['Active Ratio'])
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig('results/scaffold_target_activity_heatmap.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/scaffold_target_activity_heatmap.svg', bbox_inches='tight')
        plt.close()

    def _create_class_imbalance_visualization(self):
        """Create class imbalance analysis visualization"""
        if not hasattr(self, 'target_analysis'):
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Class Imbalance Analysis', fontsize=16, fontweight='bold')

        targets = list(self.target_analysis.keys())
        active_counts = [self.target_analysis[t]['active_compounds'] for t in targets]
        inactive_counts = [self.target_analysis[t]['inactive_compounds'] for t in targets]
        activity_ratios = [self.target_analysis[t]['activity_ratio'] for t in targets]
        imbalance_ratios = [self.target_analysis[t]['class_imbalance_ratio'] for t in targets]

        # 1. Active vs Inactive counts
        ax1 = axes[0, 0]
        x = np.arange(len(targets))
        width = 0.35

        ax1.bar(x - width/2, active_counts, width, label='Active', color='red', alpha=0.7)
        ax1.bar(x + width/2, inactive_counts, width, label='Inactive', color='blue', alpha=0.7)
        ax1.set_title('Active vs Inactive Compounds by Target')
        ax1.set_xlabel('Targets')
        ax1.set_ylabel('Number of Compounds')
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.legend()

        # 2. Activity ratios
        ax2 = axes[0, 1]
        colors = ['red' if ratio < 0.2 else 'green' if ratio > 0.8 else 'orange' for ratio in activity_ratios]
        bars = ax2.bar(targets, activity_ratios, color=colors, alpha=0.7)
        ax2.axhline(y=0.2, color='black', linestyle='--', alpha=0.5, label='20% threshold')
        ax2.axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_title('Activity Ratio by Target')
        ax2.set_xlabel('Targets')
        ax2.set_ylabel('Activity Ratio')
        ax2.set_xticklabels(targets, rotation=45, ha='right')
        ax2.legend()

        # 3. Class imbalance ratios
        ax3 = axes[1, 0]
        ax3.bar(targets, imbalance_ratios, color='purple', alpha=0.7)
        ax3.set_title('Class Imbalance Ratio (Inactive/Active)')
        ax3.set_xlabel('Targets')
        ax3.set_ylabel('Imbalance Ratio')
        ax3.set_xticklabels(targets, rotation=45, ha='right')

        # 4. Data completeness
        ax4 = axes[1, 1]
        completeness = [self.target_analysis[t]['data_completeness'] for t in targets]
        ax4.bar(targets, completeness, color='green', alpha=0.7)
        ax4.set_title('Data Completeness by Target')
        ax4.set_xlabel('Targets')
        ax4.set_ylabel('Completeness Ratio')
        ax4.set_xticklabels(targets, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('results/class_imbalance_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/class_imbalance_analysis.svg', bbox_inches='tight')
        plt.close()

    def _create_scaffold_visualizations(self):
        """Create comprehensive scaffold analysis visualizations"""
        print("\n🏗️ Creating Scaffold Analysis Visualizations...")

        if not hasattr(self, 'scaffold_df'):
            print("⚠️ Scaffold data not available. Running scaffold analysis first...")
            self._analyze_scaffold_decomposition()

        # Create scaffold diversity visualization
        self._create_scaffold_diversity_plots()

        # Create scaffold-target activity visualization
        if hasattr(self, 'scaffold_target_analysis'):
            self._create_scaffold_target_activity_plots()

        print("✅ Scaffold Analysis Visualizations created successfully!")

    def _create_scaffold_diversity_plots(self):
        """Create target-specific scaffold diversity and distribution plots"""
        # 1. Overall scaffold frequency distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Chemical Scaffold Analysis', fontsize=16, fontweight='bold')

        # Scaffold frequency histogram
        scaffold_counts = self.scaffold_df['scaffold_smiles'].value_counts()
        ax1 = axes[0, 0]
        ax1.hist(scaffold_counts.values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Scaffold Frequency Distribution')
        ax1.set_xlabel('Number of Compounds per Scaffold')
        ax1.set_ylabel('Number of Scaffolds')
        ax1.set_yscale('log')

        # Scaffold MW distribution
        ax2 = axes[0, 1]
        ax2.hist(self.scaffold_df['scaffold_mw'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Scaffold Molecular Weight Distribution')
        ax2.set_xlabel('Scaffold Molecular Weight')
        ax2.set_ylabel('Number of Scaffolds')

        # Scaffold rings distribution
        ax3 = axes[1, 0]
        ax3.hist(self.scaffold_df['scaffold_rings'], bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.set_title('Scaffold Ring Count Distribution')
        ax3.set_xlabel('Number of Rings')
        ax3.set_ylabel('Number of Scaffolds')

        # Scaffold aromatic rings distribution
        ax4 = axes[1, 1]
        ax4.hist(self.scaffold_df['scaffold_aromatic_rings'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title('Scaffold Aromatic Ring Count Distribution')
        ax4.set_xlabel('Number of Aromatic Rings')
        ax4.set_ylabel('Number of Scaffolds')

        plt.tight_layout()
        plt.savefig('results/scaffold_diversity_analysis.png', dpi=600, bbox_inches='tight')
        plt.savefig('results/scaffold_diversity_analysis.svg', bbox_inches='tight')
        plt.close()

        # 2. Target-specific scaffold analysis
        compounds = self.data['compounds']
        targets = self.data['targets']

        # Get all targets with activity data
        target_activity = {}
        for target in self.data_loader.target_columns:
            target_data = targets[target].dropna()
            if len(target_data) > 0:
                target_activity[target] = target_data.mean()

        all_targets = sorted(target_activity.items(), key=lambda x: x[1], reverse=True)

        # Create target-specific scaffold analysis
        for target, activity in all_targets[:6]:  # Top 6 targets
            target_data = targets[target].dropna()
            valid_indices = target_data.index.intersection(compounds.index)

            if len(valid_indices) > 0:
                target_values = target_data.loc[valid_indices]
                scaffold_with_target = self.scaffold_df.loc[valid_indices]

                active_mask = target_values == 1
                inactive_mask = target_values == 0

                if active_mask.sum() > 0 and inactive_mask.sum() > 0:
                    active_scaffolds = scaffold_with_target.loc[active_mask, 'scaffold_smiles']
                    inactive_scaffolds = scaffold_with_target.loc[inactive_mask, 'scaffold_smiles']

                    # Create comprehensive scaffold analysis for this target
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'Scaffold Analysis - {target}', fontsize=16, fontweight='bold')

                    # 1. Scaffold frequency by activity
                    ax1 = axes[0, 0]
                    active_counts = active_scaffolds.value_counts()
                    inactive_counts = inactive_scaffolds.value_counts()

                    ax1.hist(active_counts.values, bins=30, alpha=0.7, color='red',
                           label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                    ax1.hist(inactive_counts.values, bins=30, alpha=0.7, color='blue',
                           label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                           linestyle='--', linewidth=1)
                    ax1.set_title('Scaffold Frequency by Activity')
                    ax1.set_xlabel('Compounds per Scaffold')
                    ax1.set_ylabel('Density')
                    ax1.legend()

                    # 2. Scaffold MW by activity
                    ax2 = axes[0, 1]
                    active_mw = scaffold_with_target.loc[active_mask, 'scaffold_mw']
                    inactive_mw = scaffold_with_target.loc[inactive_mask, 'scaffold_mw']

                    ax2.hist(active_mw, bins=30, alpha=0.7, color='red',
                           label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                    ax2.hist(inactive_mw, bins=30, alpha=0.7, color='blue',
                           label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                           linestyle='--', linewidth=1)
                    ax2.set_title('Scaffold MW by Activity')
                    ax2.set_xlabel('Scaffold Molecular Weight')
                    ax2.set_ylabel('Density')
                    ax2.legend()

                    # 3. Scaffold rings by activity
                    ax3 = axes[0, 2]
                    active_rings = scaffold_with_target.loc[active_mask, 'scaffold_rings']
                    inactive_rings = scaffold_with_target.loc[inactive_mask, 'scaffold_rings']

                    ax3.hist(active_rings, bins=15, alpha=0.7, color='red',
                           label=f'Active ({active_mask.sum()})', density=True, histtype='step', linewidth=2)
                    ax3.hist(inactive_rings, bins=15, alpha=0.7, color='blue',
                           label=f'Inactive ({inactive_mask.sum()})', density=True, histtype='step',
                           linestyle='--', linewidth=1)
                    ax3.set_title('Scaffold Rings by Activity')
                    ax3.set_xlabel('Number of Rings')
                    ax3.set_ylabel('Density')
                    ax3.legend()

                    # 4. Scaffold diversity comparison
                    ax4 = axes[1, 0]
                    active_unique = active_scaffolds.nunique()
                    inactive_unique = inactive_scaffolds.nunique()
                    shared = len(set(active_scaffolds.unique()) & set(inactive_scaffolds.unique()))

                    categories = ['Active Only', 'Inactive Only', 'Shared']
                    values = [active_unique - shared, inactive_unique - shared, shared]
                    colors = ['red', 'blue', 'green']

                    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
                    ax4.set_title('Scaffold Diversity by Activity')
                    ax4.set_ylabel('Number of Unique Scaffolds')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               str(value), ha='center', va='bottom')

                    # 5. Scaffold efficiency (compounds per scaffold)
                    ax5 = axes[1, 1]
                    active_efficiency = active_mask.sum() / active_unique if active_unique > 0 else 0
                    inactive_efficiency = inactive_mask.sum() / inactive_unique if inactive_unique > 0 else 0

                    categories = ['Active', 'Inactive']
                    values = [active_efficiency, inactive_efficiency]
                    colors = ['red', 'blue']

                    bars = ax5.bar(categories, values, color=colors, alpha=0.7)
                    ax5.set_title('Compounds per Scaffold')
                    ax5.set_ylabel('Compounds per Scaffold')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{value:.1f}', ha='center', va='bottom')

                    # 6. Scaffold property correlation
                    ax6 = axes[1, 2]
                    ax6.scatter(scaffold_with_target.loc[active_mask, 'scaffold_mw'],
                              scaffold_with_target.loc[active_mask, 'scaffold_rings'],
                              c='red', alpha=0.7, s=20, label=f'Active ({active_mask.sum()})')
                    ax6.scatter(scaffold_with_target.loc[inactive_mask, 'scaffold_mw'],
                              scaffold_with_target.loc[inactive_mask, 'scaffold_rings'],
                              c='blue', alpha=0.7, s=20, label=f'Inactive ({inactive_mask.sum()})')
                    ax6.set_title('Scaffold MW vs Rings')
                    ax6.set_xlabel('Scaffold Molecular Weight')
                    ax6.set_ylabel('Number of Rings')
                    ax6.legend()

                    plt.tight_layout()
                    plt.savefig(f'results/scaffold_analysis_{target.lower().replace("-", "_")}.png',
                              dpi=600, bbox_inches='tight')
                    plt.savefig(f'results/scaffold_analysis_{target.lower().replace("-", "_")}.svg',
                              bbox_inches='tight')
                    plt.close()

    def generate_report(self):
        """Generate a comprehensive EDA report"""
        print("\n📄 Generating Enhanced EDA Report...")

        report = f"""
# Enhanced Tox21 Dataset EDA Report

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

        # Add fingerprint analysis
        if hasattr(self, 'fp_array'):
            report += f"""
### Molecular Fingerprint Analysis
- Fingerprint bits: {self.fp_array.shape[1]:,}
- Average bit density: {np.mean(np.mean(self.fp_array, axis=0)):.4f}
"""

        # Add clustering analysis
        if hasattr(self, 'cluster_results'):
            report += f"""
### Clustering Analysis
- Optimal number of clusters: {self.cluster_results['optimal_clusters']}
- Silhouette score: {self.cluster_results['silhouette_scores'].mean():.3f}
"""

        report += f"""
## Data Quality
- Valid SMILES: {len(self.data['compounds'])}/{len(self.raw_data)} compounds
- Missing target data: {(targets[self.data_loader.target_columns].isnull().sum().sum()):,} values

## Enhanced Visualizations Generated
- Molecular properties distribution
- Target distribution analysis
- Lipinski's Rule of Five analysis
- Molecular properties correlation matrix
- Target-specific molecular properties analysis
- Fingerprint analysis plots
- Clustering analysis visualizations
- Active/Inactive analysis visualizations

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open('results/enhanced_eda_report.md', 'w') as f:
            f.write(report)

        print("✅ Enhanced EDA Report generated: results/enhanced_eda_report.md")


def main():
    """Main function to run the enhanced EDA analysis"""
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)

    # Initialize and run EDA
    eda = Tox21EDAEnhanced()
    eda.load_and_analyze()
    eda.generate_report()

    print("\n🎉 Enhanced EDA Analysis Complete!")
    print("📁 Check the 'results/' directory for generated plots and report")


if __name__ == "__main__":
    main()