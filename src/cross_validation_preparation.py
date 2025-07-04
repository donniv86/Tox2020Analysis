"""
Cross-Validation Preparation Module for Tox21 Dataset

This module provides various cross-validation strategies:
1. Stratified K-Fold CV
2. Time Series Split (for temporal data)
3. Nested CV for hyperparameter tuning
4. Group K-Fold (for scaffold-based splits)
5. Repeated CV for stability assessment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold,
    RepeatedStratifiedKFold, LeaveOneGroupOut
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils.class_weight import compute_class_weight

class Tox21CrossValidation:
    """
    Cross-validation preparation for Tox21 dataset
    """

    def __init__(self, data_loader=None):
        """
        Initialize CV preparation

        Args:
            data_loader: Tox21DataLoader instance with loaded data
        """
        self.data_loader = data_loader
        self.scaler = StandardScaler()

    def prepare_stratified_kfold(self, target_idx, n_splits=5, n_repeats=1,
                                handle_imbalance=True, scale_features=True):
        """
        Prepare stratified k-fold cross-validation

        Args:
            target_idx: Index of target to prepare CV for
            n_splits: Number of CV folds
            n_repeats: Number of CV repetitions
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features

        Returns:
            Dictionary with CV splits and metadata
        """
        print(f"Preparing {n_splits}-fold stratified CV for {self.data_loader.target_names[target_idx]}")

        # Get target data
        y_target = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.data_loader.target_names[target_idx]}")

        X_valid = self.data_loader.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Initialize CV strategy
        if n_repeats > 1:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Prepare CV splits
        cv_splits = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_valid, y_valid)):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            # Scale features if requested
            if scale_features:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

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

            cv_splits.append({
                'fold': fold_idx,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'class_weights': class_weight_dict,
                'train_indices': train_idx,
                'test_indices': test_idx
            })

        return {
            'cv_splits': cv_splits,
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'target_name': self.data_loader.target_names[target_idx],
            'cv_strategy': 'stratified_kfold',
            'total_samples': len(X_valid),
            'n_features': X_valid.shape[1]
        }

    def prepare_nested_cv(self, target_idx, outer_splits=5, inner_splits=3,
                         handle_imbalance=True, scale_features=True):
        """
        Prepare nested cross-validation for hyperparameter tuning

        Args:
            target_idx: Index of target to prepare CV for
            outer_splits: Number of outer CV folds
            inner_splits: Number of inner CV folds
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features

        Returns:
            Dictionary with nested CV splits
        """
        print(f"Preparing nested CV ({outer_splits} outer, {inner_splits} inner) for {self.data_loader.target_names[target_idx]}")

        # Get target data
        y_target = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.data_loader.target_names[target_idx]}")

        X_valid = self.data_loader.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Initialize CV strategies
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

        nested_splits = []

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_valid, y_valid)):
            X_train_outer, X_test_outer = X_valid[train_idx], X_valid[test_idx]
            y_train_outer, y_test_outer = y_valid[train_idx], y_valid[test_idx]

            # Prepare inner CV splits
            inner_splits_fold = []

            for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                X_train_inner = X_train_outer[train_inner_idx]
                X_val_inner = X_train_outer[val_inner_idx]
                y_train_inner = y_train_outer[train_inner_idx]
                y_val_inner = y_train_outer[val_inner_idx]

                # Scale features if requested
                if scale_features:
                    X_train_inner = self.scaler.fit_transform(X_train_inner)
                    X_val_inner = self.scaler.transform(X_val_inner)

                # Handle class imbalance if requested
                if handle_imbalance:
                    smote = SMOTE(random_state=42)
                    X_train_inner, y_train_inner = smote.fit_resample(X_train_inner, y_train_inner)

                inner_splits_fold.append({
                    'inner_fold': inner_fold,
                    'X_train': X_train_inner,
                    'X_val': X_val_inner,
                    'y_train': y_train_inner,
                    'y_val': y_val_inner
                })

            # Scale outer test set
            if scale_features:
                X_test_outer = self.scaler.transform(X_test_outer)

            nested_splits.append({
                'outer_fold': outer_fold,
                'X_train_outer': X_train_outer,
                'X_test_outer': X_test_outer,
                'y_train_outer': y_train_outer,
                'y_test_outer': y_test_outer,
                'inner_splits': inner_splits_fold,
                'train_indices': train_idx,
                'test_indices': test_idx
            })

        return {
            'nested_splits': nested_splits,
            'outer_splits': outer_splits,
            'inner_splits': inner_splits,
            'target_name': self.data_loader.target_names[target_idx],
            'cv_strategy': 'nested_cv',
            'total_samples': len(X_valid),
            'n_features': X_valid.shape[1]
        }

    def prepare_scaffold_based_cv(self, target_idx, n_splits=5,
                                 handle_imbalance=True, scale_features=True):
        """
        Prepare scaffold-based cross-validation (groups molecules by scaffold)

        Args:
            target_idx: Index of target to prepare CV for
            n_splits: Number of CV folds
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features

        Returns:
            Dictionary with scaffold-based CV splits
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        print(f"Preparing scaffold-based CV for {self.data_loader.target_names[target_idx]}")

        # Get target data
        y_target = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.data_loader.target_names[target_idx]}")

        X_valid = self.data_loader.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Load molecules and compute scaffolds
        suppl = Chem.SDMolSupplier(self.data_loader.sdf_path)
        valid_molecules = []
        scaffolds = []

        for i, mol in enumerate(suppl):
            if mol is not None and valid_mask[i]:
                valid_molecules.append(mol)
                # Get Bemis-Murcko scaffold
                scaffold = AllChem.MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else "no_scaffold"
                scaffolds.append(scaffold_smiles)

        # Create scaffold groups
        scaffold_to_group = {}
        group_counter = 0

        for scaffold in scaffolds:
            if scaffold not in scaffold_to_group:
                scaffold_to_group[scaffold] = group_counter
                group_counter += 1

        groups = [scaffold_to_group[scaffold] for scaffold in scaffolds]

        # Initialize group-based CV
        cv = GroupKFold(n_splits=n_splits)

        # Prepare CV splits
        cv_splits = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_valid, y_valid, groups=groups)):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            # Scale features if requested
            if scale_features:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

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

            cv_splits.append({
                'fold': fold_idx,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'class_weights': class_weight_dict,
                'train_indices': train_idx,
                'test_indices': test_idx,
                'train_scaffolds': [scaffolds[i] for i in train_idx],
                'test_scaffolds': [scaffolds[i] for i in test_idx]
            })

        return {
            'cv_splits': cv_splits,
            'n_splits': n_splits,
            'target_name': self.data_loader.target_names[target_idx],
            'cv_strategy': 'scaffold_based',
            'total_samples': len(X_valid),
            'n_features': X_valid.shape[1],
            'n_scaffolds': len(set(scaffolds))
        }

    def prepare_time_series_cv(self, target_idx, n_splits=5,
                              handle_imbalance=True, scale_features=True):
        """
        Prepare time series cross-validation (for temporal data)

        Args:
            target_idx: Index of target to prepare CV for
            n_splits: Number of CV folds
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features

        Returns:
            Dictionary with time series CV splits
        """
        print(f"Preparing time series CV for {self.data_loader.target_names[target_idx]}")

        # Get target data
        y_target = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.data_loader.target_names[target_idx]}")

        X_valid = self.data_loader.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Initialize time series CV
        cv = TimeSeriesSplit(n_splits=n_splits)

        # Prepare CV splits
        cv_splits = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_valid)):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            # Scale features if requested
            if scale_features:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

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

            cv_splits.append({
                'fold': fold_idx,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'class_weights': class_weight_dict,
                'train_indices': train_idx,
                'test_indices': test_idx
            })

        return {
            'cv_splits': cv_splits,
            'n_splits': n_splits,
            'target_name': self.data_loader.target_names[target_idx],
            'cv_strategy': 'time_series',
            'total_samples': len(X_valid),
            'n_features': X_valid.shape[1]
        }

    def prepare_holdout_test_set(self, target_idx, test_size=0.2,
                                handle_imbalance=True, scale_features=True):
        """
        Prepare a single train/test split with cross-validation on training set

        Args:
            target_idx: Index of target to prepare data for
            test_size: Proportion for test set
            handle_imbalance: Whether to handle class imbalance
            scale_features: Whether to scale features

        Returns:
            Dictionary with train/test split and CV splits
        """
        from sklearn.model_selection import train_test_split

        print(f"Preparing holdout test set for {self.data_loader.target_names[target_idx]}")

        # Get target data
        y_target = self.data_loader.targets[:, target_idx]
        valid_mask = ~np.isnan(y_target)

        if not valid_mask.any():
            raise ValueError(f"No valid samples for target {self.data_loader.target_names[target_idx]}")

        X_valid = self.data_loader.descriptors[valid_mask]
        y_valid = y_target[valid_mask]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid,
            test_size=test_size,
            random_state=42,
            stratify=y_valid
        )

        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

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

        # Prepare CV splits for training set
        cv_splits = self.prepare_stratified_kfold(
            target_idx, n_splits=5, handle_imbalance=False, scale_features=False
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'class_weights': class_weight_dict,
            'cv_splits': cv_splits['cv_splits'],
            'target_name': self.data_loader.target_names[target_idx],
            'cv_strategy': 'holdout_with_cv',
            'total_samples': len(X_valid),
            'n_features': X_valid.shape[1]
        }

    def save_cv_splits(self, cv_data, filename):
        """Save CV splits to file"""
        import pickle

        with open(filename, 'wb') as f:
            pickle.dump(cv_data, f)
        print(f"Saved CV splits to {filename}")

    def load_cv_splits(self, filename):
        """Load CV splits from file"""
        import pickle

        with open(filename, 'rb') as f:
            cv_data = pickle.load(f)
        print(f"Loaded CV splits from {filename}")
        return cv_data


def main():
    """Example usage of the Tox21CrossValidation"""

    # Import data loader
    from data_preparation import Tox21DataLoader

    # Initialize data loader and load data
    loader = Tox21DataLoader()
    loader.load_descriptors()
    loader.load_targets_from_sdf()
    loader.remove_low_variance_features(threshold=0.01)
    loader.handle_missing_values(strategy='drop')

    # Initialize CV preparation
    cv_prep = Tox21CrossValidation(loader)

    # Test different CV strategies
    print("Testing different cross-validation strategies...")

    # 1. Stratified K-Fold
    print("\n1. Stratified K-Fold CV:")
    stratified_cv = cv_prep.prepare_stratified_kfold(target_idx=0, n_splits=5)
    print(f"   {stratified_cv['n_splits']} folds prepared")

    # 2. Nested CV
    print("\n2. Nested CV:")
    nested_cv = cv_prep.prepare_nested_cv(target_idx=0, outer_splits=5, inner_splits=3)
    print(f"   {nested_cv['outer_splits']} outer, {nested_cv['inner_splits']} inner folds prepared")

    # 3. Holdout with CV
    print("\n3. Holdout Test Set with CV:")
    holdout_cv = cv_prep.prepare_holdout_test_set(target_idx=0, test_size=0.2)
    print(f"   Train: {holdout_cv['X_train'].shape[0]}, Test: {holdout_cv['X_test'].shape[0]}")

    # Save CV splits
    cv_prep.save_cv_splits(stratified_cv, 'results/stratified_cv_target_0.pkl')
    cv_prep.save_cv_splits(nested_cv, 'results/nested_cv_target_0.pkl')
    cv_prep.save_cv_splits(holdout_cv, 'results/holdout_cv_target_0.pkl')


if __name__ == "__main__":
    main()