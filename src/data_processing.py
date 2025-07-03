"""
Data processing module for Tox21 dataset.
Handles loading, cleaning, and preprocessing of molecular data.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Tox21DataLoader:
    """
    Loader class for Tox21 dataset from SDF files.
    """

    def __init__(self, sdf_path: str):
        """
        Initialize the data loader.

        Args:
            sdf_path: Path to the SDF file
        """
        self.sdf_path = sdf_path
        self.data = None
        self.target_columns = [
            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from SDF file and return processed dataframes.

        Returns:
            Dictionary containing 'compounds' and 'targets' dataframes
        """
        print(f"Loading data from {self.sdf_path}...")

        # Load SDF file
        mol_df = PandasTools.LoadSDF(
            self.sdf_path,
            smilesName='SMILES',
            molColName='Molecule',
            includeFingerprints=True
        )

        print(f"Loaded {len(mol_df)} compounds")

        # Extract compound information
        compounds = self._extract_compound_info(mol_df)

        # Extract target information
        targets = self._extract_target_info(mol_df)

        # Clean data
        compounds, targets = self._clean_data(compounds, targets)

        self.data = {
            'compounds': compounds,
            'targets': targets
        }

        return self.data

    def _extract_compound_info(self, mol_df: pd.DataFrame) -> pd.DataFrame:
        """Extract compound information from the loaded dataframe."""
        compounds = pd.DataFrame()

        # Basic compound info
        compounds['compound_id'] = mol_df.get('ID', range(len(mol_df)))
        compounds['smiles'] = mol_df['SMILES']
        compounds['molecule'] = mol_df['Molecule']

        # Molecular properties
        compounds['molecular_weight'] = mol_df.get('MW', [None] * len(mol_df))
        compounds['logp'] = mol_df.get('LogP', [None] * len(mol_df))
        compounds['hbd'] = mol_df.get('HBD', [None] * len(mol_df))
        compounds['hba'] = mol_df.get('HBA', [None] * len(mol_df))
        compounds['tpsa'] = mol_df.get('TPSA', [None] * len(mol_df))
        compounds['rotatable_bonds'] = mol_df.get('RotatableBonds', [None] * len(mol_df))

        return compounds

    def _extract_target_info(self, mol_df: pd.DataFrame) -> pd.DataFrame:
        """Extract target information from the loaded dataframe."""
        targets = pd.DataFrame()

        # Extract all target columns
        for target in self.target_columns:
            if target in mol_df.columns:
                targets[target] = mol_df[target]
            else:
                targets[target] = np.nan

        return targets

    def _clean_data(self, compounds: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean and validate the data."""
        print("Cleaning data...")

        # Remove compounds with invalid SMILES
        valid_mask = compounds['smiles'].notna() & (compounds['smiles'] != '')
        compounds = compounds[valid_mask].reset_index(drop=True)
        targets = targets[valid_mask].reset_index(drop=True)

        # Convert target values to numeric, replacing non-numeric with NaN
        for col in self.target_columns:
            targets[col] = pd.to_numeric(targets[col], errors='coerce')

        # Remove compounds with no target data
        has_targets = targets[self.target_columns].notna().any(axis=1)
        compounds = compounds[has_targets].reset_index(drop=True)
        targets = targets[has_targets].reset_index(drop=True)

        print(f"After cleaning: {len(compounds)} compounds with target data")

        return compounds, targets

    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        compounds = self.data['compounds']
        targets = self.data['targets']

        summary = {
            'total_compounds': len(compounds),
            'targets': {}
        }

        for target in self.target_columns:
            target_data = targets[target].dropna()
            summary['targets'][target] = {
                'total_samples': len(target_data),
                'positive_samples': int(target_data.sum()),
                'negative_samples': int((target_data == 0).sum()),
                'positive_ratio': float(target_data.mean())
            }

        return summary

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Split data into train and test sets.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with train and test splits
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        from sklearn.model_selection import train_test_split

        compounds = self.data['compounds']
        targets = self.data['targets']

        # Split indices
        train_idx, test_idx = train_test_split(
            range(len(compounds)),
            test_size=test_size,
            random_state=random_state,
            stratify=targets[self.target_columns].sum(axis=1)  # Stratify by number of positive targets
        )

        splits = {
            'train': {
                'compounds': compounds.iloc[train_idx].reset_index(drop=True),
                'targets': targets.iloc[train_idx].reset_index(drop=True)
            },
            'test': {
                'compounds': compounds.iloc[test_idx].reset_index(drop=True),
                'targets': targets.iloc[test_idx].reset_index(drop=True)
            }
        }

        print(f"Train set: {len(splits['train']['compounds'])} compounds")
        print(f"Test set: {len(splits['test']['compounds'])} compounds")

        return splits


class DataValidator:
    """Utility class for validating molecular data."""

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """Validate if a SMILES string is valid."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @staticmethod
    def get_molecular_properties(smiles: str) -> Dict:
        """Calculate molecular properties for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        from rdkit.Chem import Descriptors, rdMolDescriptors

        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'rings': Descriptors.RingCount(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'atoms': mol.GetNumAtoms()
        }

        return properties


if __name__ == "__main__":
    # Example usage
    loader = Tox21DataLoader("data/tox21_10k_data_all.sdf")
    data = loader.load_data()

    # Print summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for target, stats in summary['targets'].items():
        print(f"{target}: {stats['positive_samples']}/{stats['total_samples']} positive ({stats['positive_ratio']:.2%})")