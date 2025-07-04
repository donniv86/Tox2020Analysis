import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    def __init__(self, correlation_threshold=0.95, univariate_k=300, top_n_model=100, random_state=42):
        self.correlation_threshold = correlation_threshold
        self.univariate_k = univariate_k
        self.top_n_model = top_n_model
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.history_ = {}

    def correlation_filter(self, X, feature_names):
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        X_corr = df.drop(columns=to_drop).values
        feature_names_corr = [f for f in feature_names if f not in to_drop]
        self.history_['correlation_filter'] = {'removed': to_drop, 'remaining': feature_names_corr}
        return X_corr, feature_names_corr

    def univariate_selection(self, X, y, feature_names):
        k = min(self.univariate_k, X.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        X_uni = selector.fit_transform(X, y)
        selected_uni = selector.get_support(indices=True)
        feature_names_uni = [feature_names[i] for i in selected_uni]
        self.history_['univariate_selection'] = {'selected': feature_names_uni}
        return X_uni, feature_names_uni

    def model_based_selection(self, X, y, feature_names):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(self.top_n_model, len(indices))
        X_final = X[:, indices[:top_n]]
        feature_names_final = [feature_names[i] for i in indices[:top_n]]
        self.feature_importances_ = importances[indices[:top_n]]
        self.selected_features_ = feature_names_final
        self.history_['model_based_selection'] = {'selected': feature_names_final, 'importances': self.feature_importances_}
        return X_final, feature_names_final

    def fit_transform(self, X, y, feature_names):
        # Step 1: Correlation filter
        X_corr, feature_names_corr = self.correlation_filter(X, feature_names)
        # Step 2: Univariate selection
        X_uni, feature_names_uni = self.univariate_selection(X_corr, y, feature_names_corr)
        # Step 3: Model-based selection
        X_final, feature_names_final = self.model_based_selection(X_uni, y, feature_names_uni)
        self.feature_names_ = feature_names_final
        return X_final, feature_names_final

    def plot_feature_importances(self, save_path=None, top_n=None):
        if self.feature_importances_ is None or self.selected_features_ is None:
            print("Run fit_transform first.")
            return
        n = top_n if top_n is not None else len(self.selected_features_)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.feature_importances_[:n], y=self.selected_features_[:n])
        plt.title('Top Feature Importances (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage
def example_usage():
    from src.data_preparation import Tox21DataLoader
    loader = Tox21DataLoader()
    loader.load_descriptors()
    loader.load_targets_from_sdf()
    loader.remove_low_variance_features(threshold=0.01)
    loader.handle_missing_values(strategy='drop')
    target_idx = 0  # NR-AR
    y = loader.targets[:, target_idx]
    X = loader.descriptors
    feature_names = loader.feature_names
    selector = FeatureSelector(correlation_threshold=0.95, univariate_k=300, top_n_model=100)
    X_selected, selected_names = selector.fit_transform(X, y, feature_names)
    print(f"Selected features shape: {X_selected.shape}")
    selector.plot_feature_importances(save_path='results/NR-AR_feature_importances_class.png')
    np.save('results/NR-AR_selected_features_class.npy', X_selected)
    np.save('results/NR-AR_selected_feature_names_class.npy', selected_names)

if __name__ == "__main__":
    example_usage()