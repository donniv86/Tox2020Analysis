# Tox21 Modeling Pipeline - Modular Structure

This document describes the clean, modular structure of the Tox21 modeling pipeline.

## 🏗️ Architecture Overview

The pipeline has been refactored into a clean, modular structure with the following components:

```
tox21_models/
├── src/
│   ├── pipeline_manager.py    # Main orchestrator
│   ├── model_trainer.py       # Model training and evaluation
│   ├── result_manager.py      # Result saving/loading
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── feature_selector.py    # Feature selection
│   └── ... (other modules)
├── config.py                  # Centralized configuration
├── run_pipeline.py           # Simple main script
├── utils.py                  # Utility functions
└── MODULAR_STRUCTURE.md      # This file
```

## 🚀 Quick Start

### 1. Run the Pipeline
```bash
python run_pipeline.py
```

### 2. Use Utility Functions
```bash
python utils.py
```

## 📋 Core Components

### 1. PipelineManager (`src/pipeline_manager.py`)
**Main orchestrator** that coordinates the entire workflow.

**Key Features:**
- Data loading and preprocessing
- Feature selection
- Model training and evaluation
- Progress tracking and logging
- Resume capability

**Usage:**
```python
from src.pipeline_manager import PipelineManager
from config import PIPELINE_CONFIG

pipeline = PipelineManager(PIPELINE_CONFIG)
results = pipeline.run_pipeline(target_indices=[0, 7])
```

### 2. ModelTrainer (`src/model_trainer.py`)
**Handles model training and evaluation** with cross-validation.

**Key Features:**
- Multiple model types (RandomForest, LogisticRegression, SVM)
- Cross-validation training
- Performance evaluation
- Result aggregation

**Usage:**
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(
    models=['RandomForest', 'LogisticRegression'],
    cv_folds=5,
    random_state=42
)
results = trainer.train_and_evaluate(X, y, target_name)
```

### 3. ResultManager (`src/result_manager.py`)
**Manages saving and loading** of results, models, and reports.

**Key Features:**
- Save/load trained models
- Save/load feature selectors
- Generate comparison reports
- Export results for sharing

**Usage:**
```python
from src.result_manager import ResultManager

result_manager = ResultManager('results')
result_manager.save_target_results(target_name, results, feature_selector, selected_features)
```

### 4. Tox21Utils (`utils.py`)
**Utility functions** for common operations.

**Key Features:**
- Load trained models
- Make predictions on new compounds
- Compare target performance
- Generate prediction reports

**Usage:**
```python
from utils import Tox21Utils

utils = Tox21Utils()
model_data = utils.load_model_and_features('NR-AR')
predictions = utils.predict_toxicity('NR-AR', descriptors, feature_names)
```

## ⚙️ Configuration

All settings are centralized in `config.py`:

```python
PIPELINE_CONFIG = {
    'random_state': 42,
    'cv_folds': 5,
    'models': ['RandomForest', 'LogisticRegression', 'SVM'],
    'feature_selection': {
        'correlation_threshold': 0.90,
        'univariate_k': 500,
        'top_n_model': 150
    },
    # ... more settings
}
```

## 📊 Output Structure

```
results/
├── {target}_results.pkl           # Detailed training results
├── {target}_best_model.pkl        # Best trained model
├── {target}_feature_selector.pkl  # Feature selector
├── {target}_selected_features.npy # Selected feature names
├── {target}_model_comparison.csv  # Model comparison report
└── pipeline_summary.csv           # Overall summary

logs/
├── pipeline_{timestamp}.log       # Detailed execution log
└── pipeline_progress.json         # Progress tracking
```

## 🔧 Maintenance Benefits

### 1. **Separation of Concerns**
- Each class has a single responsibility
- Easy to modify individual components
- Clear interfaces between modules

### 2. **Easy Debugging**
- Comprehensive logging at each step
- Isolated components for testing
- Clear error messages and stack traces

### 3. **Configuration Management**
- All settings in one place
- Easy to experiment with different parameters
- Environment-specific configurations

### 4. **Reusability**
- Components can be used independently
- Easy to extend with new models/features
- Clean APIs for integration

### 5. **Progress Tracking**
- Automatic progress saving
- Resume capability after interruption
- Detailed execution logs

## 🛠️ Common Operations

### Add a New Model
1. Add model configuration to `config.py`
2. Update `ModelTrainer._get_model_configs()`
3. No changes needed in other components

### Change Feature Selection
1. Modify settings in `config.py`
2. Or update `PIPELINE_CONFIG['feature_selection']`
3. Pipeline automatically uses new settings

### Process Different Targets
```python
# Process specific targets
results = pipeline.run_pipeline(target_indices=[0, 1, 2])

# Process all targets
results = pipeline.run_pipeline()
```

### Load and Use Trained Models
```python
from utils import Tox21Utils

utils = Tox21Utils()
model_data = utils.load_model_and_features('NR-AR')

# Make predictions
predictions = utils.predict_toxicity('NR-AR', new_descriptors, feature_names)
```

## 🐛 Debugging

### Check Logs
```bash
tail -f logs/pipeline_*.log
```

### Verify Results
```bash
python utils.py  # Lists available targets and performance
```

### Resume Interrupted Run
```bash
python run_pipeline.py  # Automatically resumes from last saved state
```

## 📈 Performance Monitoring

The pipeline provides comprehensive performance tracking:

- **Real-time logging** with timestamps
- **Progress tracking** with resume capability
- **Performance metrics** for each model
- **Summary reports** for easy comparison
- **Export functionality** for sharing results

## 🔄 Extending the Pipeline

### Add New Models
1. Update `ModelTrainer._get_model_configs()`
2. Add model to `PIPELINE_CONFIG['models']`

### Add New Evaluation Metrics
1. Update `ModelTrainer._calculate_metrics()`
2. Add metric to `EVALUATION_METRICS` in `config.py`

### Add New Feature Selection Methods
1. Extend `FeatureSelector` class
2. Update configuration as needed

This modular structure makes the codebase **maintainable**, **debuggable**, and **extensible** while providing a clean, professional interface for Tox21 modeling.