"""
Data Preparation Module for Tox21 Dataset

This module handles:
1. Loading descriptors and target labels
2. Data splitting (train/validation/test)
3. Class imbalance handling
4. Basic data preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class Tox21DataLoader:
    """
    Data loader for Tox21 dataset with comprehensive preprocessing
    """

    def __init__(self, descriptors_path='results/tox21_descriptors.npy',
                 sdf_path='data/tox21_10k_data_all.sdf'):
        """
        Initialize the data loader

        Args:
            descriptors_path: Path to the generated descriptors
            sdf_path: Path to the original SDF file with target labels
        """
        self.descriptors_path = descriptors_path
        self.sdf_path = sdf_path
        self.descriptors = None
        self.targets = None
        self.feature_names = None
        self.target_names = None
        self.scaler = StandardScaler()

    def load_descriptors(self):
        """Load the generated descriptors"""
        print("Loading descriptors...")
        self.descriptors = np.load(self.descriptors_path)
        print(f"Loaded descriptors shape: {self.descriptors.shape}")

        # Generate feature names
        self.feature_names = [f"desc_{i}" for i in range(self.descriptors.shape[1])]
        return self.descriptors

    def load_targets_from_sdf(self):
        """Load target labels from SDF file"""
        from rdkit import Chem

        print("Loading target labels from SDF...")

        # Define Tox21 target names
        self.target_names = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]

        # Read SDF file and extract targets
        suppl = Chem.SDMolSupplier(self.sdf_path)

        # Initialize lists to store valid molecules and their targets
        valid_molecules = []
        valid_targets = []

        for i, mol in enumerate(suppl):
            if mol is not None:
                # Initialize target vector for this molecule
                mol_targets = np.full(len(self.target_names), np.nan)

                for j, target_name in enumerate(self.target_names):
                    # Try different possible property names
                    prop_names = [target_name, f'Activity_{target_name}',
                                f'TOX21_{target_name}_active']

                    value = None
                    for prop_name in prop_names:
                        if mol.HasProp(prop_name):
                            value = mol.GetProp(prop_name)
                            break

                    if value is not None:
                        # Convert to binary (1 for active, 0 for inactive)
                        if value.lower() in ['1', 'active', 'true', 'yes']:
                            mol_targets[j] = 1
                        elif value.lower() in ['0', 'inactive', 'false', 'no']:
                            mol_targets[j] = 0

                # Only keep molecules that have at least one valid target
                if not np.isnan(mol_targets).all():
                    valid_molecules.append(i)
                    valid_targets.append(mol_targets)

        # Convert to numpy array
        self.targets = np.array(valid_targets)

        # Filter descriptors to match valid molecules
        if len(valid_molecules) != self.descriptors.shape[0]:
            print(f"Filtering descriptors: {self.descriptors.shape[0]} -> {len(valid_molecules)} molecules")
            self.descriptors = self.descriptors[valid_molecules]

        print(f"Loaded targets shape: {self.targets.shape}")
        print(f"Valid molecules: {len(valid_molecules)}")
        return self.targets

    def get_target_statistics(self):
        """Get statistics for each target"""
        if self.targets is None:
            raise ValueError("Targets not loaded yet. Call load_targets_from_sdf() first.")

        stats = []
        for i, target_name in enumerate(self.target_names):
            target_data = self.targets[:, i]
            valid_mask = ~np.isnan(target_data)
            valid_data = target_data[valid_mask]

            if len(valid_data) > 0:
                n_active = np.sum(valid_data == 1)
                n_inactive = np.sum(valid_data == 0)
                n_total = len(valid_data)
                active_ratio = n_active / n_total if n_total > 0 else 0

                stats.append({
                    'target': target_name,
                    'total_samples': n_total,
                    'active_samples': n_active,
                    'inactive_samples': n_inactive,
                    'active_ratio': active_ratio,
                    'imbalance_ratio': n_inactive / n_active if n_active > 0 else float('inf')
                })

        return pd.DataFrame(stats)

    def handle_infinite_values(self):
        """Handle infinite values in descriptors"""
        print("Handling infinite values...")

        # Replace infinite values with NaN
        self.descriptors = np.where(np.isinf(self.descriptors), np.nan, self.descriptors)

        # Count infinite values
        n_inf = np.sum(np.isinf(self.descriptors))
        print(f"Found {n_inf} infinite values")

        # Replace NaN with median of each feature
        for i in range(self.descriptors.shape[1]):
            col = self.descriptors[:, i]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                if not np.isnan(median_val):
                    col = np.nan_to_num(col, nan=median_val)
                else:
                    # If all values are NaN, replace with 0
                    col = np.nan_to_num(col, nan=0)
                self.descriptors[:, i] = col

        print("✓ Handled infinite and NaN values")
        return self.descriptors

    def remove_low_variance_features(self, threshold=0.01):
        """Remove features with low variance"""
        from sklearn.feature_selection import VarianceThreshold

        print(f"Removing features with variance < {threshold}...")

        # Handle infinite values first
        self.handle_infinite_values()

        selector = VarianceThreshold(threshold=threshold)
        self.descriptors = selector.fit_transform(self.descriptors)

        # Update feature names
        self.feature_names = [f"desc_{i}" for i in range(self.descriptors.shape[1])]

        print(f"Features after variance thresholding: {self.descriptors.shape[1]}")
        return self.descriptors

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values in targets

        Args:
            strategy: 'drop' to remove samples with missing values,
                     'impute' to fill with most frequent value
        """
        if strategy == 'drop':
            # Remove samples with any missing target values
            valid_mask = ~np.isnan(self.targets).any(axis=1)
            self.descriptors = self.descriptors[valid_mask]
            self.targets = self.targets[valid_mask]
            print(f"After removing missing values: {self.descriptors.shape[0]} samples")

        elif strategy == 'impute':
            # Fill missing values with most frequent value (0 for inactive)
            self.targets = np.nan_to_num(self.targets, nan=0)
            print("Filled missing values with 0 (inactive)")

    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train/validation/test sets

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed for reproducibility
        """
        print("Splitting data into train/validation/test sets...")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.descriptors, self.targets,
            test_size=test_size,
            random_state=random_state,
            stratify=self.targets[:, 0] if not np.isnan(self.targets[:, 0]).all() else None
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp[:, 0] if not np.isnan(y_temp[:, 0]).all() else None
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("Scaling features...")

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_class_weights(self, y_train):
        """Compute class weights for imbalanced datasets"""
        print("Computing class weights...")

        class_weights = {}
        for i, target_name in enumerate(self.target_names):
            y_target = y_train[:, i]
            valid_mask = ~np.isnan(y_target)
            y_valid = y_target[valid_mask]

            if len(y_valid) > 0 and len(np.unique(y_valid)) > 1:
                weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(y_valid),
                    y=y_valid
                )
                class_weights[target_name] = dict(zip(np.unique(y_valid), weights))
            else:
                class_weights[target_name] = {0: 1.0, 1: 1.0}

        return class_weights

    def apply_smote(self, X_train, y_train, target_idx=0):
        """
        Apply SMOTE for handling class imbalance

        Args:
            X_train: Training features
            y_train: Training targets
            target_idx: Index of target to balance
        """
        print(f"Applying SMOTE for target {self.target_names[target_idx]}...")

        y_target = y_train[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            print("No valid samples for this target")
            return X_train, y_train

        X_valid = X_train[valid_mask]
        y_valid = y_target[valid_mask]

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_valid, y_valid)

        # Update the target column
        y_train_resampled = y_train.copy()
        y_train_resampled[valid_mask, target_idx] = y_resampled

        print(f"Original samples: {len(X_valid)}")
        print(f"Resampled samples: {len(X_resampled)}")

        return X_resampled, y_train_resampled

    def prepare_data_for_target(self, target_idx, test_size=0.2, val_size=0.2,
                               handle_imbalance=True, scale_features=True):
        """
        Prepare data for a specific target

        Args:
            target_idx: Index of target to prepare data for
            test_size: Proportion for test set
            val_size: Proportion for validation set
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features
        """
        print(f"Preparing data for target: {self.target_names[target_idx]}")

        # Get target data
        y_target = self.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.target_names[target_idx]}")

        X_valid = self.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_valid, y_valid,
            test_size=test_size,
            random_state=42,
            stratify=y_valid
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )

        # Scale features if requested
        if scale_features:
            X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # Handle class imbalance if requested
        if handle_imbalance:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Class weights: {class_weight_dict}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'class_weights': class_weight_dict,
            'target_name': self.target_names[target_idx]
        }

    def get_data_summary(self):
        """Get comprehensive data summary"""
        summary = {
            'descriptors_shape': self.descriptors.shape if self.descriptors is not None else None,
            'targets_shape': self.targets.shape if self.targets is not None else None,
            'n_features': self.descriptors.shape[1] if self.descriptors is not None else None,
            'n_samples': self.descriptors.shape[0] if self.descriptors is not None else None,
            'n_targets': len(self.target_names) if self.target_names is not None else None,
            'target_names': self.target_names,
            'feature_names': self.feature_names[:10] + ['...'] if self.feature_names else None
        }

        if self.targets is not None:
            target_stats = self.get_target_statistics()
            summary['target_statistics'] = target_stats

        return summary


def main():
    """Example usage of the Tox21DataLoader"""

    # Initialize data loader
    loader = Tox21DataLoader()

    # Load data
    loader.load_descriptors()
    loader.load_targets_from_sdf()

    # Get target statistics
    target_stats = loader.get_target_statistics()
    print("\nTarget Statistics:")
    print(target_stats)

    # Remove low variance features
    loader.remove_low_variance_features(threshold=0.01)

    # Handle missing values
    loader.handle_missing_values(strategy='drop')

    # Get data summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        if key != 'target_statistics':
            print(f"{key}: {value}")

    # Prepare data for first target
    data_dict = loader.prepare_data_for_target(
        target_idx=0,
        handle_imbalance=True,
        scale_features=True
    )

    print(f"\nPrepared data for {data_dict['target_name']}")
    print(f"Training samples: {data_dict['X_train'].shape[0]}")
    print(f"Class weights: {data_dict['class_weights']}")


if __name__ == "__main__":
    main()