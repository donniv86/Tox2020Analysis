# Tox21 Toxicity Prediction Models

A comprehensive machine learning pipeline for predicting toxicity endpoints using the Tox21 dataset.

## Overview

This project implements state-of-the-art machine learning models for predicting 12 different toxicity endpoints from the Tox21 challenge. The Tox21 dataset contains ~10,000 compounds with experimental toxicity data across multiple biological pathways.

## Toxicity Endpoints

1. **NR-Aromatase** - Nuclear Receptor Aromatase
2. **NR-AR** - Nuclear Receptor Androgen Receptor
3. **NR-AR-LBD** - Nuclear Receptor Androgen Receptor Ligand Binding Domain
4. **NR-ER** - Nuclear Receptor Estrogen Receptor
5. **NR-ER-LBD** - Nuclear Receptor Estrogen Receptor Ligand Binding Domain
6. **NR-PPAR-gamma** - Nuclear Receptor Peroxisome Proliferator-Activated Receptor Gamma
7. **NR-AhR** - Nuclear Receptor Aryl Hydrocarbon Receptor
8. **SR-ARE** - Stress Response Antioxidant Response Element
9. **SR-ATAD5** - Stress Response ATAD5
10. **SR-HSE** - Stress Response Heat Shock Element
11. **SR-MMP** - Stress Response Mitochondrial Membrane Potential
12. **SR-p53** - Stress Response p53

## Project Structure

```
tox21_models/
├── data/                          # Data files
│   └── tox21_10k_data_all.sdf    # Original Tox21 dataset
├── src/                           # Source code
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── feature_engineering.py    # Molecular fingerprint generation
│   ├── models.py                 # ML model implementations
│   ├── evaluation.py             # Model evaluation metrics
│   └── visualization.py          # Plotting and visualization
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── models/                        # Trained model files
├── results/                       # Results and outputs
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tox21_models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For RDKit installation issues on macOS:
```bash
conda install -c conda-forge rdkit
```

## Usage

### Quick Start

```python
from src.data_processing import Tox21DataLoader
from src.feature_engineering import MolecularFeatureGenerator
from src.models import ToxicityPredictor

# Load data
loader = Tox21DataLoader('data/tox21_10k_data_all.sdf')
data = loader.load_data()

# Generate features
feature_gen = MolecularFeatureGenerator()
features = feature_gen.generate_features(data['smiles'])

# Train model
predictor = ToxicityPredictor()
predictor.train(features, data['targets'])

# Make predictions
predictions = predictor.predict(new_smiles)
```

### Step-by-Step Tutorial

1. **Data Exploration**: Run `notebooks/01_data_exploration.ipynb`
2. **Feature Engineering**: Run `notebooks/02_feature_engineering.ipynb`
3. **Model Training**: Run `notebooks/03_model_training.ipynb`
4. **Model Evaluation**: Run `notebooks/04_model_evaluation.ipynb`

## Features

- **Multiple Molecular Fingerprints**: Morgan, MACCS, RDKit, Mordred descriptors
- **Advanced ML Models**: Random Forest, XGBoost, Neural Networks, Graph Neural Networks
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Balanced Accuracy, Confusion Matrices
- **Interactive Visualizations**: Compound structure viewing, performance plots
- **Model Interpretability**: SHAP values, feature importance analysis

## Results

The models achieve the following performance metrics (averaged across all endpoints):

- **Random Forest**: ROC-AUC = 0.78
- **XGBoost**: ROC-AUC = 0.81
- **Neural Network**: ROC-AUC = 0.79
- **Graph Neural Network**: ROC-AUC = 0.83

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## References

- Tox21 Challenge: https://tripod.nih.gov/tox21/
- RDKit: https://www.rdkit.org/
- Mordred: https://github.com/mordred-descriptor/mordred