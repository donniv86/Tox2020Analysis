import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data_preparation import Tox21DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed data
loader = Tox21DataLoader()
loader.load_descriptors()
loader.load_targets_from_sdf()
loader.remove_low_variance_features(threshold=0.01)
loader.handle_missing_values(strategy='drop')

# 2. Select target (e.g., NR-AR)
target_idx = 0  # NR-AR
y = loader.targets[:, target_idx]
X = loader.descriptors
feature_names = loader.feature_names

# 3. Remove highly correlated features
def correlation_filter(X, feature_names, threshold=0.95):
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Removing {len(to_drop)} highly correlated features (>{threshold})")
    return df.drop(columns=to_drop).values, [f for f in feature_names if f not in to_drop]

X_corr, feature_names_corr = correlation_filter(X, feature_names, threshold=0.95)

# 4. Univariate feature selection (mutual information)
k = min(300, X_corr.shape[1])
selector = SelectKBest(mutual_info_classif, k=k)
X_uni = selector.fit_transform(X_corr, y)
selected_uni = selector.get_support(indices=True)
feature_names_uni = [feature_names_corr[i] for i in selected_uni]
print(f"Selected top {k} features by mutual information.")

# 5. Model-based feature importance (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_uni, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
# Select top 100 features
top_n = min(100, len(indices))
X_final = X_uni[:, indices[:top_n]]
feature_names_final = [feature_names_uni[i] for i in indices[:top_n]]
print(f"Selected top {top_n} features by Random Forest importance.")

# 6. Save reduced feature set
np.save('results/NR-AR_selected_features.npy', X_final)
np.save('results/NR-AR_selected_feature_names.npy', feature_names_final)
print(f"Saved reduced feature set: {X_final.shape}")

# 7. Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:top_n]], y=[feature_names_uni[i] for i in indices[:top_n]])
plt.title('Top Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/NR-AR_feature_importances.png')
plt.show()