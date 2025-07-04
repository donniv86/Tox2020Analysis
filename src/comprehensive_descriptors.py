"""
Comprehensive molecular descriptor generation for Tox21 modeling.
Generates fingerprints, RDKit descriptors, and target-specific features.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, PandasTools
import warnings
from typing import List, Dict, Optional, Tuple
import os
import time

warnings.filterwarnings('ignore')


class ComprehensiveDescriptorGenerator:
    """
    Generate comprehensive molecular descriptors for Tox21 modeling.
    Includes fingerprints, RDKit descriptors, and target-specific features.
    """

    def __init__(self,
                 use_morgan: bool = True,
                 use_maccs: bool = True,
                 use_atom_pairs: bool = True,
                 use_rdkit: bool = True,
                 use_fragments: bool = True,
                 use_target_specific: bool = True):
        """
        Initialize the descriptor generator.

        Args:
            use_morgan: Generate Morgan fingerprints (ECFP4)
            use_maccs: Generate MACCS keys
            use_atom_pairs: Generate atom pair fingerprints
            use_rdkit: Generate RDKit 2D descriptors
            use_fragments: Generate molecular fragment descriptors
            use_target_specific: Generate target-specific descriptors
        """
        self.use_morgan = use_morgan
        self.use_maccs = use_maccs
        self.use_atom_pairs = use_atom_pairs
        self.use_rdkit = use_rdkit
        self.use_fragments = use_fragments
        self.use_target_specific = use_target_specific

        self.feature_names = []
        self.descriptor_stats = {}
        self.target_columns = [
            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

    def generate_morgan_fingerprints(self, smiles_list: List[str],
                                   radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """
        Generate Morgan fingerprints (ECFP4 equivalent).

        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint

        Returns:
            Array of Morgan fingerprints
        """
        print(f"  Generating Morgan fingerprints (radius={radius}, n_bits={n_bits})...")
        fingerprints = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(n_bits))

        return np.array(fingerprints)

    def generate_maccs_keys(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate MACCS structural keys (167 bits).

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of MACCS fingerprints
        """
        print("  Generating MACCS keys...")
        fingerprints = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(167))

        return np.array(fingerprints)

    def generate_atom_pair_fingerprints(self, smiles_list: List[str],
                                      n_bits: int = 2048) -> np.ndarray:
        """
        Generate atom pair fingerprints.

        Args:
            smiles_list: List of SMILES strings
            n_bits: Number of bits in fingerprint

        Returns:
            Array of atom pair fingerprints
        """
        print(f"  Generating atom pair fingerprints (n_bits={n_bits})...")
        fingerprints = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(n_bits))

        return np.array(fingerprints)

    def generate_rdkit_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive RDKit 2D descriptors.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame of RDKit descriptors
        """
        print("  Generating RDKit descriptors...")
        descriptors = []

        # Define key RDKit descriptors for toxicity prediction
        key_descriptors = {
            # Basic properties
            'MolWt': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'TPSA': Descriptors.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            # Lipinski's Rule of Five
            'Lipinski_HBA': Descriptors.NumHAcceptors,
            'Lipinski_HBD': Descriptors.NumHDonors,
            'Lipinski_LogP': Descriptors.MolLogP,
            'Lipinski_MW': Descriptors.MolWt,
            # Topological descriptors
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT,
            # Connectivity descriptors
            'Chi0': Descriptors.Chi0,
            'Chi0n': Descriptors.Chi0n,
            'Chi0v': Descriptors.Chi0v,
            'Chi1': Descriptors.Chi1,
            'Chi1n': Descriptors.Chi1n,
            'Chi1v': Descriptors.Chi1v,
            'Chi2n': Descriptors.Chi2n,
            'Chi2v': Descriptors.Chi2v,
            'Chi3n': Descriptors.Chi3n,
            'Chi3v': Descriptors.Chi3v,
            'Chi4n': Descriptors.Chi4n,
            'Chi4v': Descriptors.Chi4v,
            # Constitutional descriptors
            'RingCount': Descriptors.RingCount,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'NumSaturatedRings': Descriptors.NumSaturatedRings,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'NumSpiroAtoms': Descriptors.NumSpiroAtoms,
            'NumBridgeheadAtoms': Descriptors.NumBridgeheadAtoms,
            # Geometric descriptors
            'FractionCSP3': Descriptors.FractionCSP3,
            'HallKierAlpha': Descriptors.HallKierAlpha,
            'Ipc': Descriptors.Ipc,
            # Electronic descriptors
            'MaxEStateIndex': Descriptors.MaxEStateIndex,
            'MinEStateIndex': Descriptors.MinEStateIndex,
            'MaxPartialCharge': Descriptors.MaxPartialCharge,
            'MinPartialCharge': Descriptors.MinPartialCharge,
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge,
            'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge,
        }

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = {}
                # Add NumAtoms and NumHeavyAtoms using mol methods
                desc['NumAtoms'] = mol.GetNumAtoms()
                desc['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
                # Calculate key descriptors
                for name, func in key_descriptors.items():
                    try:
                        desc[name] = func(mol)
                    except:
                        desc[name] = np.nan
                # Add all available RDKit descriptors
                for name, func in Descriptors.descList:
                    if name not in desc:  # Avoid duplicates
                        try:
                            desc[name] = func(mol)
                        except:
                            desc[name] = np.nan
                descriptors.append(desc)
            else:
                # Create empty descriptor dict for invalid molecules
                empty_desc = {'NumAtoms': np.nan, 'NumHeavyAtoms': np.nan}
                for name in key_descriptors.keys():
                    empty_desc[name] = np.nan
                for name, _ in Descriptors.descList:
                    empty_desc[name] = np.nan
                descriptors.append(empty_desc)

        return pd.DataFrame(descriptors)

    def generate_molecular_fragments(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Generate molecular fragment descriptors.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame of fragment descriptors
        """
        print("  Generating molecular fragments...")
        fragments = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fragment_dict = {}

                # Ring systems
                ring_info = mol.GetRingInfo()
                fragment_dict['num_rings'] = ring_info.NumRings()
                fragment_dict['num_aromatic_rings'] = sum(1 for ring in ring_info.AtomRings()
                                                        if all(mol.GetAtomWithIdx(idx).GetIsAromatic()
                                                              for idx in ring))
                fragment_dict['num_saturated_rings'] = fragment_dict['num_rings'] - fragment_dict['num_aromatic_rings']

                # Functional groups
                fragment_dict['num_oh'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]')))
                fragment_dict['num_nh'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH]')))
                fragment_dict['num_co'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=O')))
                fragment_dict['num_cn'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#N')))
                fragment_dict['num_cc_triple'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#C')))
                fragment_dict['num_cc_double'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C')))
                fragment_dict['num_aromatic'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('a')))

                # Halogens
                fragment_dict['num_f'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[F]')))
                fragment_dict['num_cl'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[Cl]')))
                fragment_dict['num_br'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[Br]')))
                fragment_dict['num_i'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[I]')))

                # Sulfur and nitrogen
                fragment_dict['num_s'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[S]')))
                fragment_dict['num_n'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N]')))

                # Molecular complexity
                fragment_dict['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                fragment_dict['num_stereocenters'] = len(Chem.FindMolChiralCenters(mol))

                fragments.append(fragment_dict)
            else:
                fragments.append({})

        return pd.DataFrame(fragments)

    def generate_target_specific_descriptors(self, smiles_list: List[str],
                                           targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate target-specific descriptors based on EDA insights.

        Args:
            smiles_list: List of SMILES strings
            targets: DataFrame with target information

        Returns:
            DataFrame of target-specific descriptors
        """
        print("  Generating target-specific descriptors...")
        target_descriptors = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(smiles_list)} molecules...")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = {}

                # Nuclear Receptor (NR) specific descriptors
                nr_descriptors = {
                    'nr_aromatic_ratio': Descriptors.FractionCSP3(mol),
                    'nr_polar_surface_area': Descriptors.TPSA(mol),
                    'nr_hydrogen_bonding': Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol),
                    'nr_molecular_weight': Descriptors.MolWt(mol),
                    'nr_logp': Descriptors.MolLogP(mol),
                    'nr_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'nr_aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'nr_heteroatoms': Descriptors.NumHeteroatoms(mol),
                }

                # Stress Response (SR) specific descriptors
                sr_descriptors = {
                    'sr_rings': Descriptors.RingCount(mol),
                    'sr_aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'sr_saturated_rings': Descriptors.NumSaturatedRings(mol),
                    'sr_molecular_weight': Descriptors.MolWt(mol),
                    'sr_logp': Descriptors.MolLogP(mol),
                    'sr_tpsa': Descriptors.TPSA(mol),
                    'sr_hbd': Descriptors.NumHDonors(mol),
                    'sr_hba': Descriptors.NumHAcceptors(mol),
                    'sr_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'sr_fraction_csp3': Descriptors.FractionCSP3(mol),
                    'sr_heteroatoms': Descriptors.NumHeteroatoms(mol),
                    'sr_spiro_atoms': Descriptors.NumSpiroAtoms(mol),
                    'sr_bridgehead_atoms': Descriptors.NumBridgeheadAtoms(mol)
                }

                # Scaffold-based descriptors
                scaffold_descriptors = {
                    'scaffold_flexibility': Descriptors.NumRotatableBonds(mol) / max(1, mol.GetNumHeavyAtoms()),
                    'scaffold_complexity': Descriptors.RingCount(mol) / max(1, mol.GetNumHeavyAtoms()),
                    'scaffold_aromaticity': Descriptors.NumAromaticRings(mol) / max(1, Descriptors.RingCount(mol)),
                    'scaffold_saturation': Descriptors.NumSaturatedRings(mol) / max(1, Descriptors.RingCount(mol)),
                    'scaffold_heteroatom_ratio': Descriptors.NumHeteroatoms(mol) / max(1, mol.GetNumHeavyAtoms()),
                    'scaffold_spiro_density': Descriptors.NumSpiroAtoms(mol) / max(1, mol.GetNumHeavyAtoms()),
                    'scaffold_bridgehead_density': Descriptors.NumBridgeheadAtoms(mol) / max(1, mol.GetNumHeavyAtoms())
                }

                # Combine all target-specific descriptors
                desc.update(nr_descriptors)
                desc.update(sr_descriptors)
                desc.update(scaffold_descriptors)

                # Add target-specific features if targets are provided
                if targets is not None and i < len(targets):
                    target_row = targets.iloc[i]
                    for target_col in self.target_columns:
                        if target_col in target_row and pd.notna(target_row[target_col]):
                            desc[f'target_{target_col}_value'] = target_row[target_col]
                        else:
                            desc[f'target_{target_col}_value'] = 0

                target_descriptors.append(desc)
            else:
                target_descriptors.append({})

        return pd.DataFrame(target_descriptors)

    def generate_all_descriptors(self, smiles_list: List[str],
                                targets: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate all molecular descriptors.

        Args:
            smiles_list: List of SMILES strings
            targets: DataFrame with target information (optional)

        Returns:
            Array of all descriptors
        """
        print(f"Generating comprehensive molecular descriptors for {len(smiles_list)} molecules...")
        start_time = time.time()

        features_list = []

        # 1. Molecular Fingerprints
        if self.use_morgan:
            morgan_fp = self.generate_morgan_fingerprints(smiles_list)
            features_list.append(morgan_fp)
            self.feature_names.extend([f'morgan_{i}' for i in range(morgan_fp.shape[1])])
            print(f"    Morgan fingerprints: {morgan_fp.shape[1]} features")

        if self.use_maccs:
            maccs_fp = self.generate_maccs_keys(smiles_list)
            features_list.append(maccs_fp)
            self.feature_names.extend([f'maccs_{i}' for i in range(maccs_fp.shape[1])])
            print(f"    MACCS keys: {maccs_fp.shape[1]} features")

        if self.use_atom_pairs:
            atom_pair_fp = self.generate_atom_pair_fingerprints(smiles_list)
            features_list.append(atom_pair_fp)
            self.feature_names.extend([f'atom_pair_{i}' for i in range(atom_pair_fp.shape[1])])
            print(f"    Atom pair fingerprints: {atom_pair_fp.shape[1]} features")

        # 2. RDKit Descriptors
        if self.use_rdkit:
            rdkit_desc = self.generate_rdkit_descriptors(smiles_list)
            rdkit_array = rdkit_desc.fillna(0).values
            features_list.append(rdkit_array)
            self.feature_names.extend(rdkit_desc.columns.tolist())
            print(f"    RDKit descriptors: {rdkit_array.shape[1]} features")

        # 3. Fragment Descriptors
        if self.use_fragments:
            fragment_desc = self.generate_molecular_fragments(smiles_list)
            fragment_array = fragment_desc.fillna(0).values
            features_list.append(fragment_array)
            self.feature_names.extend(fragment_desc.columns.tolist())
            print(f"    Fragment descriptors: {fragment_array.shape[1]} features")

        # 4. Target-Specific Descriptors
        if self.use_target_specific:
            target_desc = self.generate_target_specific_descriptors(smiles_list, targets)
            target_array = target_desc.fillna(0).values
            features_list.append(target_array)
            self.feature_names.extend(target_desc.columns.tolist())
            print(f"    Target-specific descriptors: {target_array.shape[1]} features")

        # Combine all features
        X = np.hstack(features_list)

        elapsed_time = time.time() - start_time
        print(f"\nDescriptor generation completed in {elapsed_time:.2f} seconds")
        print(f"Generated {X.shape[1]} descriptors for {X.shape[0]} molecules")

        # Store descriptor statistics
        self.descriptor_stats = {
            'n_molecules': X.shape[0],
            'n_descriptors': X.shape[1],
            'feature_names': self.feature_names,
            'sparsity': np.sum(X == 0) / X.size,
            'missing_values': np.sum(np.isnan(X)),
            'generation_time': elapsed_time
        }

        return X

    def get_descriptor_summary(self) -> Dict:
        """Get summary of generated descriptors."""
        return self.descriptor_stats

    def save_descriptors(self, X: np.ndarray, filepath: str):
        """Save descriptors to file."""
        np.save(filepath, X)
        print(f"Descriptors saved to {filepath}")

    def load_descriptors(self, filepath: str) -> np.ndarray:
        """Load descriptors from file."""
        X = np.load(filepath)
        print(f"Descriptors loaded from {filepath}: {X.shape}")
        return X


def main():
    """Example usage of the descriptor generator."""

    # Example SMILES
    example_smiles = [
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "CC1=C(C(=CC=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    ]

    # Initialize descriptor generator
    generator = ComprehensiveDescriptorGenerator(
        use_morgan=True,
        use_maccs=True,
        use_atom_pairs=True,
        use_rdkit=True,
        use_fragments=True,
        use_target_specific=True
    )

    # Generate descriptors
    X = generator.generate_all_descriptors(example_smiles)

    # Print summary
    summary = generator.get_descriptor_summary()
    print("\nDescriptor Summary:")
    for key, value in summary.items():
        if key != 'feature_names':
            print(f"  {key}: {value}")

    print(f"\nFeature names: {len(summary['feature_names'])} features")
    print(f"Descriptor matrix shape: {X.shape}")


if __name__ == "__main__":
    main()