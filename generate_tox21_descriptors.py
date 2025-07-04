"""
Script to generate comprehensive molecular descriptors for the Tox21 dataset.
Loads the SDF, computes descriptors, and saves them for modeling.
"""

from src.data_processing import Tox21DataLoader
from src.comprehensive_descriptors import ComprehensiveDescriptorGenerator
import os

# Path to SDF file
SDF_PATH = "data/tox21_10k_data_all.sdf"
# Output path
OUTPUT_PATH = "results/tox21_descriptors.npy"


def main():
    print("Loading Tox21 data from SDF...")
    loader = Tox21DataLoader(SDF_PATH)
    data = loader.load_data()
    compounds = data['compounds']
    targets = data['targets']
    smiles_list = compounds['smiles'].tolist()

    print(f"Loaded {len(smiles_list)} molecules. Generating descriptors...")
    generator = ComprehensiveDescriptorGenerator(
        use_morgan=True,
        use_maccs=True,
        use_atom_pairs=True,
        use_rdkit=True,
        use_fragments=True,
        use_target_specific=True
    )
    X = generator.generate_all_descriptors(smiles_list, targets=targets)

    # Print summary
    summary = generator.get_descriptor_summary()
    print("\nDescriptor Summary:")
    for key, value in summary.items():
        if key != 'feature_names':
            print(f"  {key}: {value}")
    print(f"\nFeature names: {len(summary['feature_names'])} features")
    print(f"Descriptor matrix shape: {X.shape}")

    # Save descriptors
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    generator.save_descriptors(X, OUTPUT_PATH)
    print(f"Descriptors saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()