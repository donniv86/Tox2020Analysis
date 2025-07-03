"""
Feature engineering module for molecular fingerprints and descriptors.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class MolecularFeatureGenerator:
    """Generate molecular fingerprints and descriptors."""

    def __init__(self):
        self.feature_names = []

    def generate_morgan_fingerprints(self, smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Generate Morgan fingerprints."""
        fingerprints = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(n_bits))

        return np.array(fingerprints)

    def generate_maccs_keys(self, smiles_list: List[str]) -> np.ndarray:
        """Generate MACCS keys."""
        fingerprints = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(167))

        return np.array(fingerprints)

    def generate_rdkit_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """Generate RDKit descriptors."""
        descriptors = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = {}
                for name, func in Descriptors.descList:
                    try:
                        desc[name] = func(mol)
                    except:
                        desc[name] = np.nan
                descriptors.append(desc)
            else:
                descriptors.append({name: np.nan for name, _ in Descriptors.descList})

        return pd.DataFrame(descriptors)

    def generate_features(self, smiles_list: List[str],
                         use_morgan: bool = True,
                         use_maccs: bool = True,
                         use_descriptors: bool = True) -> np.ndarray:
        """Generate all molecular features."""
        features_list = []

        if use_morgan:
            morgan_fp = self.generate_morgan_fingerprints(smiles_list)
            features_list.append(morgan_fp)
            self.feature_names.extend([f'morgan_{i}' for i in range(morgan_fp.shape[1])])

        if use_maccs:
            maccs_fp = self.generate_maccs_keys(smiles_list)
            features_list.append(maccs_fp)
            self.feature_names.extend([f'maccs_{i}' for i in range(maccs_fp.shape[1])])

        if use_descriptors:
            desc_df = self.generate_rdkit_descriptors(smiles_list)
            desc_array = desc_df.fillna(0).values
            features_list.append(desc_array)
            self.feature_names.extend(desc_df.columns.tolist())

        return np.hstack(features_list)