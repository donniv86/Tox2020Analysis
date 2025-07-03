"""
Machine learning models for toxicity prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ToxicityPredictor:
    """Main class for toxicity prediction models."""

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.models = {}
        self.target_columns = [
            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

    def _get_model(self, target: str):
        """Get model instance for a specific target."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            return SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: pd.DataFrame, targets: Optional[List[str]] = None):
        """Train models for all targets."""
        if targets is None:
            targets = self.target_columns

        for target in targets:
            if target in y.columns:
                print(f"Training model for {target}...")
                model = self._get_model(target)

                # Get target data
                target_data = y[target].dropna()
                target_indices = target_data.index
                X_target = X[target_indices]
                y_target = target_data.values

                # Train model
                model.fit(X_target, y_target)
                self.models[target] = model

    def predict(self, X: np.ndarray, targets: Optional[List[str]] = None) -> pd.DataFrame:
        """Make predictions for all targets."""
        if targets is None:
            targets = list(self.models.keys())

        predictions = {}
        for target in targets:
            if target in self.models:
                pred_proba = self.models[target].predict_proba(X)[:, 1]
                predictions[target] = pred_proba

        return pd.DataFrame(predictions)

    def evaluate(self, X: np.ndarray, y: pd.DataFrame) -> Dict:
        """Evaluate model performance."""
        results = {}

        for target in self.target_columns:
            if target in y.columns and target in self.models:
                target_data = y[target].dropna()
                target_indices = target_data.index
                X_target = X[target_indices]
                y_target = target_data.values

                # Predictions
                y_pred_proba = self.models[target].predict_proba(X_target)[:, 1]
                y_pred = self.models[target].predict(X_target)

                # Metrics
                roc_auc = roc_auc_score(y_target, y_pred_proba)
                pr_auc = average_precision_score(y_target, y_pred_proba)

                results[target] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }

        return results

    def save_models(self, path: str):
        """Save trained models."""
        joblib.dump(self.models, path)

    def load_models(self, path: str):
        """Load trained models."""
        self.models = joblib.load(path)


class EnsemblePredictor:
    """Ensemble of multiple models."""

    def __init__(self, models: List[str] = ['random_forest', 'logistic', 'svm']):
        self.models = models
        self.predictors = {}
        self.target_columns = [
            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

    def train(self, X: np.ndarray, y: pd.DataFrame):
        """Train ensemble of models."""
        for model_type in self.models:
            print(f"Training {model_type}...")
            predictor = ToxicityPredictor(model_type)
            predictor.train(X, y)
            self.predictors[model_type] = predictor

    def predict(self, X: np.ndarray, method: str = 'average') -> pd.DataFrame:
        """Make ensemble predictions."""
        predictions = {}

        for target in self.target_columns:
            target_preds = []
            for model_type, predictor in self.predictors.items():
                if target in predictor.models:
                    pred = predictor.predict(X)[target]
                    target_preds.append(pred)

            if target_preds:
                if method == 'average':
                    predictions[target] = np.mean(target_preds, axis=0)
                elif method == 'max':
                    predictions[target] = np.max(target_preds, axis=0)
                elif method == 'min':
                    predictions[target] = np.min(target_preds, axis=0)

        return pd.DataFrame(predictions)