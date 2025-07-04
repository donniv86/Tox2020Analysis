"""
Advanced machine learning models for Tox21 toxicity prediction.
Incorporates insights from EDA analysis for target-specific modeling.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
import joblib
import warnings
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


class AdvancedToxicityPredictor:
    """
    Advanced toxicity predictor with target-specific modeling and ensemble methods.
    Incorporates insights from EDA analysis.
    """

    def __init__(self,
                 model_type: str = 'ensemble',
                 feature_selection: bool = True,
                 use_scaling: bool = True,
                 target_specific: bool = True):
        """
        Initialize the advanced predictor.

        Args:
            model_type: Type of model ('ensemble', 'rf', 'gbm', 'svm', 'mlp', 'xgboost', 'lightgbm')
            feature_selection: Whether to perform feature selection
            use_scaling: Whether to scale features
            target_specific: Whether to use target-specific hyperparameters
        """
        self.model_type = model_type
        self.feature_selection = feature_selection
        self.use_scaling = use_scaling
        self.target_specific = target_specific

        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}

        # Target-specific configurations based on EDA insights
        self.target_configs = {
            'NR-Aromatase': {
                'class_weight': 'balanced',
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10
            },
            'NR-AR': {
                'class_weight': 'balanced',
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 8
            },
            'NR-AR-LBD': {
                'class_weight': 'balanced',
                'n_estimators': 180,
                'max_depth': 14,
                'min_samples_split': 9
            },
            'NR-ER': {
                'class_weight': 'balanced',
                'n_estimators': 160,
                'max_depth': 13,
                'min_samples_split': 7
            },
            'NR-ER-LBD': {
                'class_weight': 'balanced',
                'n_estimators': 170,
                'max_depth': 14,
                'min_samples_split': 8
            },
            'NR-PPAR-gamma': {
                'class_weight': 'balanced',
                'n_estimators': 140,
                'max_depth': 11,
                'min_samples_split': 6
            },
            'NR-AhR': {
                'class_weight': 'balanced',
                'n_estimators': 190,
                'max_depth': 16,
                'min_samples_split': 10
            },
            'SR-ARE': {
                'class_weight': 'balanced',
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 12
            },
            'SR-ATAD5': {
                'class_weight': 'balanced',
                'n_estimators': 130,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'SR-HSE': {
                'class_weight': 'balanced',
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 7
            },
            'SR-MMP': {
                'class_weight': 'balanced',
                'n_estimators': 160,
                'max_depth': 13,
                'min_samples_split': 8
            },
            'SR-p53': {
                'class_weight': 'balanced',
                'n_estimators': 180,
                'max_depth': 14,
                'min_samples_split': 9
            }
        }

    def _get_model(self, target: str, X_shape: Tuple[int, int]):
        """Get model instance for a specific target."""
        n_features = X_shape[1]

        if self.target_specific and target in self.target_configs:
            config = self.target_configs[target]
        else:
            config = {
                'class_weight': 'balanced',
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 8
            }

        if self.model_type == 'ensemble':
            # Create ensemble of different models
            models = [
                ('rf', RandomForestClassifier(
                    n_estimators=config['n_estimators'],
                    max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'],
                    class_weight=config['class_weight'],
                    random_state=42
                )),
                ('gbm', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )),
                ('xgb', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ))
            ]
            return VotingClassifier(estimators=models, voting='soft')

        elif self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                class_weight=config['class_weight'],
                random_state=42
            )

        elif self.model_type == 'gbm':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )

        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

        elif self.model_type == 'svm':
            return SVC(
                probability=True,
                class_weight='balanced',
                random_state=42
            )

        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_features(self, X: np.ndarray, target: str, is_training: bool = True):
        """Prepare features with scaling and selection."""
        X_processed = X.copy()

        # Scaling
        if self.use_scaling:
            if is_training:
                if self.model_type in ['svm', 'mlp']:
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
                X_processed = scaler.fit_transform(X_processed)
                self.scalers[target] = scaler
            else:
                if target in self.scalers:
                    X_processed = self.scalers[target].transform(X_processed)

        # Feature selection
        if self.feature_selection and is_training:
            n_features = min(1000, X_processed.shape[1] // 2)
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_processed = selector.fit_transform(X_processed, y=None)  # y will be passed later
            self.feature_selectors[target] = selector
        elif self.feature_selection and not is_training and target in self.feature_selectors:
            X_processed = self.feature_selectors[target].transform(X_processed)

        return X_processed

    def train(self, X: np.ndarray, y: pd.DataFrame, targets: Optional[List[str]] = None):
        """Train models for all targets."""
        if targets is None:
            targets = y.columns.tolist()

        print(f"Training {self.model_type} models for {len(targets)} targets...")

        for i, target in enumerate(targets):
            if target in y.columns:
                print(f"[{i+1}/{len(targets)}] Training model for {target}...")

                # Get target data
                target_data = y[target].dropna()
                target_indices = target_data.index
                X_target = X[target_indices]
                y_target = target_data.values

                # Skip if not enough data
                if len(y_target) < 50:
                    print(f"  Skipping {target}: insufficient data ({len(y_target)} samples)")
                    continue

                # Prepare features
                X_processed = self._prepare_features(X_target, target, is_training=True)

                # Get model
                model = self._get_model(target, X_processed.shape)

                # Train model
                model.fit(X_processed, y_target)
                self.models[target] = model

                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[target] = model.feature_importances_
                elif hasattr(model, 'named_estimators_') and 'rf' in model.named_estimators_:
                    self.feature_importance[target] = model.named_estimators_['rf'].feature_importances_

                print(f"  Completed training for {target}")

    def predict(self, X: np.ndarray, targets: Optional[List[str]] = None) -> pd.DataFrame:
        """Make predictions for all targets."""
        if targets is None:
            targets = list(self.models.keys())

        predictions = {}
        for target in targets:
            if target in self.models:
                # Prepare features
                X_processed = self._prepare_features(X, target, is_training=False)

                # Make predictions
                pred_proba = self.models[target].predict_proba(X_processed)[:, 1]
                predictions[target] = pred_proba

        return pd.DataFrame(predictions)

    def evaluate(self, X: np.ndarray, y: pd.DataFrame) -> Dict:
        """Evaluate model performance with detailed metrics."""
        results = {}

        for target in y.columns:
            if target in self.models:
                target_data = y[target].dropna()
                target_indices = target_data.index
                X_target = X[target_indices]
                y_target = target_data.values

                # Prepare features
                X_processed = self._prepare_features(X_target, target, is_training=False)

                # Predictions
                y_pred_proba = self.models[target].predict_proba(X_processed)[:, 1]
                y_pred = self.models[target].predict(X_processed)

                # Metrics
                roc_auc = roc_auc_score(y_target, y_pred_proba)
                pr_auc = average_precision_score(y_target, y_pred_proba)

                # Cross-validation
                cv_scores = cross_val_score(
                    self.models[target], X_processed, y_target,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc'
                )

                results[target] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'n_samples': len(y_target),
                    'n_positive': int(y_target.sum()),
                    'n_negative': int((y_target == 0).sum())
                }

        return results

    def get_feature_importance(self, target: str, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance for a specific target."""
        if target not in self.feature_importance:
            return pd.DataFrame()

        importance = self.feature_importance[target]

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def save_models(self, path: str):
        """Save trained models and preprocessing objects."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'target_configs': self.target_configs
        }
        joblib.dump(model_data, path)

    def load_models(self, path: str):
        """Load trained models and preprocessing objects."""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_selectors = model_data['feature_selectors']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data.get('model_type', 'ensemble')
        self.target_configs = model_data.get('target_configs', {})


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""

    def __init__(self):
        self.results = {}

    def evaluate_model(self, model: AdvancedToxicityPredictor, X: np.ndarray, y: pd.DataFrame) -> Dict:
        """Evaluate a model and store results."""
        results = model.evaluate(X, y)
        self.results[model.model_type] = results
        return results

    def compare_models(self) -> pd.DataFrame:
        """Compare performance across different models."""
        comparison_data = []

        for model_type, results in self.results.items():
            for target, metrics in results.items():
                comparison_data.append({
                    'model_type': model_type,
                    'target': target,
                    'roc_auc': metrics['roc_auc'],
                    'pr_auc': metrics['pr_auc'],
                    'cv_mean': metrics['cv_mean'],
                    'cv_std': metrics['cv_std'],
                    'n_samples': metrics['n_samples'],
                    'n_positive': metrics['n_positive'],
                    'n_negative': metrics['n_negative']
                })

        return pd.DataFrame(comparison_data)

    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Plot performance comparison across models and targets."""
        df = self.compare_models()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC AUC comparison
        sns.boxplot(data=df, x='model_type', y='roc_auc', ax=axes[0,0])
        axes[0,0].set_title('ROC AUC by Model Type')
        axes[0,0].set_ylabel('ROC AUC')

        # PR AUC comparison
        sns.boxplot(data=df, x='model_type', y='pr_auc', ax=axes[0,1])
        axes[0,1].set_title('PR AUC by Model Type')
        axes[0,1].set_ylabel('PR AUC')

        # Target-wise performance
        target_perf = df.groupby('target')['roc_auc'].mean().sort_values(ascending=False)
        target_perf.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Average ROC AUC by Target')
        axes[1,0].set_ylabel('ROC AUC')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Sample size vs performance
        axes[1,1].scatter(df['n_samples'], df['roc_auc'], alpha=0.6)
        axes[1,1].set_xlabel('Number of Samples')
        axes[1,1].set_ylabel('ROC AUC')
        axes[1,1].set_title('Performance vs Sample Size')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class HyperparameterOptimizer:
    """Hyperparameter optimization for target-specific models."""

    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.best_params = {}

    def optimize_hyperparameters(self, X: np.ndarray, y: pd.Series, target: str) -> Dict:
        """Optimize hyperparameters for a specific target."""
        print(f"Optimizing hyperparameters for {target}...")

        if self.model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)

        elif self.model_type == 'gbm':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = GradientBoostingClassifier(random_state=42)

        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = xgb.XGBClassifier(random_state=42)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1
        )

        grid_search.fit(X, y)

        self.best_params[target] = grid_search.best_params_

        print(f"Best parameters for {target}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_