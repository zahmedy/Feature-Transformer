# Feature Transformer

[![Tests](https://github.com/zahmedy/Feature-Transformer/actions/workflows/tests.yml/badge.svg)](https://github.com/zahmedy/Feature-Transformer/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready feature engineering library that eliminates boilerplate in tabular ML projects. Built on scikit-learn's `ColumnTransformer` with intelligent defaults for preprocessing pipelines.

## Key Features

- **Zero-Config Intelligence**: Automatic detection of numeric and categorical columns with support for explicit schema definition
- **Production-Ready Preprocessing**: Built-in imputation, scaling, and one-hot encoding with configurable strategies
- **Feature Selection**: Integrated SelectKBest for both classification and regression tasks
- **Pipeline Compatible**: Seamlessly integrates with scikit-learn pipelines and cross-validation
- **Type-Safe**: Full type hints and dataclass-based configuration
- **Test Coverage**: Comprehensive test suite including edge cases, invariants, schema drift detection, and serialization

## Installation

```bash
# Clone the repository
git clone https://github.com/zayed/Feature-Transformer.git
cd Feature-Transformer

# Install in development mode
pip install -e ".[dev]"

# Or install without dev dependencies
pip install -e .
```

## Quick Start

### Basic Usage with Auto-Detection

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src import FeatureTransformer

# Load your data
df = pd.read_csv("data/raw/customer_segments.csv")
X = df.drop(columns=["cluster_id"])
y = df["cluster_id"]

# Create pipeline with auto-detection
pipeline = Pipeline([
    ("features", FeatureTransformer(task="classification")),
    ("clf", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)
predictions = pipeline.predict(X)
```

### Advanced Usage with Custom Configuration

```python
from src import FeatureTransformer, FeatureConfig

# Define explicit feature schema
config = FeatureConfig(
    numeric_features=["age", "income", "tenure"],
    categorical_features=["segment", "region"]
)

# Configure preprocessing with feature selection
transformer = FeatureTransformer(
    feature_config=config,
    task="classification",
    k_best=10,                    # Select top 10 features
    scale_numeric=True,           # StandardScaler for numerics
    handle_unknown="ignore"       # Ignore unknown categories
)

# Fit and transform
X_transformed = transformer.fit_transform(X, y)

# Access feature names
feature_names = transformer.get_feature_names_out()
print(f"Selected features: {feature_names}")
```

### Regression Tasks

```python
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ("features", FeatureTransformer(
        task="regression",
        k_best=15,
        scale_numeric=True
    )),
    ("model", Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## API Reference

### `FeatureTransformer`

**Parameters:**
- `feature_config` (FeatureConfig, optional): Explicit column specification. If `None`, auto-detects from DataFrame dtypes.
- `task` (Literal["classification", "regression"]): Task type for feature selection scoring function.
- `k_best` (int, optional): Number of top features to select. If `None`, no feature selection is applied.
- `scale_numeric` (bool): Whether to apply StandardScaler to numeric features. Default: `True`.
- `handle_unknown` (str): Strategy for unknown categories in one-hot encoding. Default: `"ignore"`.

**Methods:**
- `fit(X, y)`: Fit the transformer on training data
- `transform(X)`: Transform data using fitted transformer
- `fit_transform(X, y)`: Fit and transform in one call
- `get_feature_names_out()`: Return transformed feature names

### `FeatureConfig`

**Parameters:**
- `numeric_features` (Sequence[str], optional): List of numeric column names
- `categorical_features` (Sequence[str], optional): List of categorical column names

## Project Structure

```
Feature-Transformer/
├── .github/
│   └── workflows/
│       └── tests.yml              # CI/CD pipeline
├── data/
│   └── raw/
│       └── customer_segments.csv  # Sample dataset
├── src/
│   ├── __init__.py
│   └── transformer.py             # Core transformer implementation
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_smoke.py              # Basic functionality tests
│   ├── test_edge_cases.py         # Edge case handling
│   ├── test_invariants.py         # Property-based testing
│   ├── test_schema_drift.py       # Schema validation
│   └── test_serialization.py     # Model persistence
├── utils/
│   └── run.py                     # Demo script
├── pyproject.toml                 # Package configuration
└── requirements.txt               # Dependencies
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_smoke.py -v
```

### Running the Demo

```bash
python utils/run.py
```

The demo script trains a classifier on sample customer segmentation data and displays:
- Train/test accuracy scores
- Transformed feature names
- Feature selection results (if enabled)

## Architecture diagram
```mermaid
flowchart TD
  A[Raw pandas DataFrame X] --> B[FeatureTransformer.fit<X, y>]
  B --> C[Build column schema<br/>• explicit FeatureConfig<br/>• dtype auto-detect fallback]
  C --> D[ColumnTransformer]
  D --> N[Numeric pipeline<br/>Imputer → Scaler <optional>]
  D --> K[Categorical pipeline<br/>Imputer → OneHotEncoder<handle_unknown=ignore>]
  N --> E[Concatenate features]
  K --> E
  E --> F{SelectKBest?}
  F -- No --> G[ndarray features Xt]
  F -- Yes --> H[SelectKBest<br/><f_classif / f_regression>]
  H --> G
  G --> I[Downstream sklearn Estimator]
  I --> J[Predictions]

  subgraph Production Guarantees <tested>
    P1[Missing columns → ValueError <strict schema>]
    P2[Unseen categories → no crash + fixed feature dimension]
    P3[No NaNs after transform]
    P4[Column order invariance]
    P5[joblib round-trip identical output]
  end
  ```


## Technical Highlights

### Preprocessing Pipeline
- **Numeric Features**: Median imputation → StandardScaler (optional)
- **Categorical Features**: Most-frequent imputation → OneHotEncoder
- **Feature Selection**: SelectKBest with F-statistics (f_classif/f_regression)

### Robustness
- Validates input is pandas DataFrame
- Checks for missing columns in explicit configuration
- Handles scikit-learn version compatibility (sparse vs sparse_output parameter)
- Proper fit/transform separation prevents data leakage

### Testing Strategy
- **Smoke Tests**: Core functionality validation
- **Edge Cases**: Empty data, single columns, all-null scenarios
- **Invariants**: Property-based testing with Hypothesis
- **Schema Drift**: Runtime validation of expected columns
- **Serialization**: Pickle compatibility for model persistence

## Use Cases

- **Rapid Prototyping**: Eliminate preprocessing boilerplate in Jupyter notebooks
- **Production Pipelines**: Drop-in transformer for production ML systems
- **AutoML Components**: Reusable preprocessing module for AutoML frameworks
- **Education**: Learn scikit-learn best practices with working examples

## Dependencies

- Python 3.9+
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Author

**Zayed Ahmed**

## Acknowledgments

Built with scikit-learn's robust preprocessing infrastructure and inspired by common patterns in production ML systems.
