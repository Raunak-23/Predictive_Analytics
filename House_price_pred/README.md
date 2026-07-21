 # House Price Prediction & Valuation: A Comparative Machine Learning Approach

This project contains a comprehensive machine learning workflow evaluating and comparing various regression algorithms on two distinct real estate datasets: the **UCI Real Estate Valuation** dataset and the **Ames Housing** dataset.

Using regression models ranging from simple linear baselines to advanced tuned ensemble methods (Random Forest, Gradient Boosting, XGBoost), the notebooks detail steps for data preprocessing, exploratory data analysis, robust cross-validation, hyperparameter optimization, and model serialization.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Datasets Overview](#datasets-overview)
3. [Jupyter Notebook Workflows](#jupyter-notebook-workflows)
4. [Verified Model Results](#verified-model-results)
   - [UCI Real Estate Valuation Results](#uci-real-estate-valuation-results)
   - [Ames Housing Results](#ames-housing-results)
5. [Reproducing the Results](#reproducing-the-results)
6. [Serialized Model Artifacts](#serialized-model-artifacts)
7. [CLI Tool for Inference & Retraining](#cli-tool-for-inference--retraining)
   - [Commands Overview](#commands-overview)
   - [Prediction Examples](#prediction-examples)
   - [Retraining Examples](#retraining-examples)
   - [Listing Models](#listing-models)
   - [Interactive Mode](#interactive-mode)
8. [Environment & Dependencies](#environment--dependencies)

---

## Project Structure

```
House_price_pred/
├── AmesHousing.csv                    # Dataset for Ames Housing (2,930 rows, 82 columns)
├── ames_housing.ipynb                 # Jupyter Notebook for Ames Housing dataset
├── uci_house.ipynb                    # Jupyter Notebook for UCI Real Estate Valuation
├── uci_house.html                     # Exported HTML of UCI notebook
├── uci_all_model_results.csv          # Evaluation summary for all UCI models (including tuned)
├── uci_cv_results.csv                 # 5-fold cross-validation metrics for UCI models
├── uci_evaluation_results.csv         # Test-split evaluation metrics for default UCI models
├── house_price_cli.py                 # CLI tool for inference, retraining & model management
├── artifacts/                         # Serialization directory for models, pipelines & metadata
│   ├── ames_all_results.csv           # Evaluation summary for all Ames models
│   ├── ames_cv_results.csv            # 5-fold cross-validation metrics for Ames models
│   ├── ames_metadata.json             # Ames dataset training run metadata
│   ├── ames_best_model.joblib         # Serialized best Ames model (XGBoost)
│   ├── ames_preprocessor.joblib       # Serialized Ames preprocessing pipeline
│   ├── uci_best_model.joblib          # Serialized best UCI model (Tuned Random Forest)
│   ├── uci_preprocessing_pipeline.joblib # Serialized UCI preprocessing pipeline
│   ├── uci_model_registry.json        # Detailed registry of UCI model parameters
│   ├── uci_training_metadata.json     # UCI training metadata
│   └── uci_[model_name].joblib        # Serialized estimators for each model
├── parse_notebooks.py                 # Utility script to analyze notebook structure
└── README.md                          # Project Documentation (this file)
```

---

## Datasets Overview

### 1. UCI Real Estate Valuation Dataset
* **Origin**: Collected from Sindian District, New Taipei City, Taiwan (compiled by I-Cheng Yeh).
* **Dataset Scale**: 414 instances, 6 features, 1 target variable.
* **Target**: `Y house price of unit area` (measured in units of 10,000 New Taiwan Dollar/Ping, where 1 Ping = 3.3 square meters).
* **Explanatory Features**:
  1. `X1 transaction date` (e.g., 2013.250 = 2013 March)
  2. `X2 house age` (unit: years)
  3. `X3 distance to the nearest MRT station` (unit: meters)
  4. `X4 number of convenience stores` (integer count in walking distance)
  5. `X5 latitude` (geographic coordinate in degrees)
  6. `X6 longitude` (geographic coordinate in degrees)

### 2. Ames Housing Dataset
* **Origin**: Compiled by Dean De Cock, documenting house sales in Ames, Iowa, USA from 2006 to 2010.
* **Dataset Scale**: 2,930 instances, 79 explanatory features, 1 target variable.
* **Target**: `SalePrice` (measured in USD).
* **Explanatory Features**: 79 columns encompassing nominal, ordinal, discrete, and continuous variables describing property quality, condition, zoning, location, dimensions, basement, garage, and facilities.

---

## Jupyter Notebook Workflows

The notebooks [uci_house.ipynb](file:///c:/Users/apara/Downloads/VIT%20Downloads/Predictive%20lab/House_price_pred/uci_house.ipynb) and [ames_housing.ipynb](file:///c:/Users/apara/Downloads/VIT%20Downloads/Predictive%20lab/House_price_pred/ames_housing.ipynb) implement a robust Scikit-Learn based machine learning pipeline:

1. **Environment Setup & Imports**: Standardizes package imports and sets a fixed seed (`SEED = 42`) to ensure reproducibility.
2. **Data Auditing**: Checks dataset shape, data types, missing values, duplicates, and verifies target distributions.
3. **Train-Test Split**: Implements a clean 80/20 train-test split before doing any exploratory data analysis or pipeline fitting to prevent data leakage.
4. **Exploratory Data Analysis (EDA)**: Conducted on the training split only. Generates histograms for target variables and scatter plots of features against the target (e.g., living area vs price).
5. **Feature Preprocessing Pipeline**:
   * **Numeric Columns**: Missing values are imputed using median (or mean) strategy, followed by standardization using `StandardScaler`.
   * **Categorical Columns** (Ames only): Missing values are filled with the most frequent category, and nominal values are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to safely handle unseen classes in test data.
   * All preprocessors are packaged into a modular `ColumnTransformer` object.
6. **Naive Baseline Model**: Uses a `DummyRegressor(strategy='mean')` to establish a performance floor.
7. **Simple Linear Regression**: Trains on a single, highly correlated feature (e.g., `Gr Liv Area` for Ames, `X3 distance to the nearest MRT station` for UCI) to establish a baseline interpretable model.
8. **OLS & Regularized Linear Models**: Fits Multiple Linear Regression, Ridge, Lasso, and Elastic Net to evaluate linear baselines and feature shrinkage.
9. **Non-Linear Tree & Ensemble Regressors**: Trains Decision Trees, Random Forests, Gradient Boosting, and XGBoost models.
10. **Hyperparameter Tuning**: Utilizes `GridSearchCV` on the training set (with 3-fold or 5-fold cross-validation) to optimize parameters for the top-performing regressors.
11. **Final Evaluation**: Computes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and $R^2$ on the held-out test split.
12. **Serialization & Asset Savings**: Uses `joblib` to serialize the trained preprocessing pipeline, the best estimator, all individual models, and writes out final evaluation summaries as CSVs and JSON metadata.

---

## Verified Model Results

All results listed below are loaded and verified directly from the serialized CSV outputs in the directory.

### UCI Real Estate Valuation Results

#### 1. 5-Fold Cross-Validation Performance (Training Set)
*Models sorted by mean Cross-Validation RMSE (lowest to highest).*

| Rank | Model | CV MAE (Mean) | CV MAE (Std) | CV RMSE (Mean) | CV RMSE (Std) | CV $R^2$ (Mean) | CV $R^2$ (Std) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost** | 5.11 | 0.62 | 7.71 | 2.06 | 0.68 | 0.08 |
| 2 | **Random Forest** | 5.22 | 0.74 | 7.78 | 2.27 | 0.67 | 0.11 |
| 3 | **Gradient Boosting** | 5.44 | 0.38 | 8.35 | 1.82 | 0.62 | 0.06 |
| 4 | **Ridge Regression** | 6.55 | 0.78 | 9.11 | 2.26 | 0.55 | 0.10 |
| 5 | **Elastic Net** | 6.55 | 0.78 | 9.11 | 2.25 | 0.55 | 0.10 |
| 6 | **Lasso Regression** | 6.55 | 0.78 | 9.11 | 2.25 | 0.55 | 0.10 |
| 7 | **Linear Regression** | 6.55 | 0.78 | 9.11 | 2.25 | 0.55 | 0.10 |
| 8 | **Decision Tree** | 6.12 | 0.58 | 9.58 | 2.26 | 0.50 | 0.13 |

#### 2. Final Test Set Performance
*Sorted by Test RMSE (lowest to highest). Naive baseline Mean RMSE is 13.06.*

| Model | Test MAE | Test RMSE | Test $R^2$ | MAPE (%) | Train MAE | Train RMSE | Train $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned Random Forest** | **3.7342** | **5.5174** | **0.8185** | **11.61** | - | - | - |
| XGBoost | 3.7615 | 5.5327 | 0.8175 | 12.31 | 3.4306 | 4.8544 | 0.8747 |
| Tuned XGBoost | 3.8926 | 5.6601 | 0.8090 | 12.47 | - | - | - |
| Gradient Boosting | 3.9036 | 5.8443 | 0.7964 | 11.96 | 2.6921 | 3.6284 | 0.9300 |
| Random Forest | 4.2806 | 5.9220 | 0.7910 | 13.64 | 4.3557 | 6.3443 | 0.7860 |
| Decision Tree | 4.7395 | 6.2470 | 0.7674 | 14.68 | 4.6826 | 6.3676 | 0.7844 |
| Polynomial Regression | 4.9593 | 6.8856 | 0.7174 | 16.49 | - | - | - |
| Tuned Ridge | 5.2969 | 7.2835 | 0.6838 | 16.92 | - | - | - |
| Ridge Regression | 5.3022 | 7.3108 | 0.6814 | 17.15 | 6.3404 | 9.1167 | 0.5581 |
| Elastic Net | 5.3048 | 7.3139 | 0.6811 | 17.18 | 6.3399 | 9.1167 | 0.5581 |
| Lasso Regression | 5.3052 | 7.3144 | 0.6811 | 17.18 | 6.3398 | 9.1167 | 0.5581 |
| Multiple Linear Regression | 5.3054 | 7.3148 | 0.6811 | 17.18 | 6.3397 | 9.1167 | 0.5581 |
| Simple Linear Regression | 6.9210 | 8.7945 | 0.5390 | 23.90 | - | - | - |

> [!NOTE]
> **Best UCI Model**: **Tuned Random Forest** with hyperparameters: `n_estimators=200`, `max_depth=10`, `min_samples_leaf=2`. It achieved the lowest Test RMSE of **5.5174** and an $R^2$ of **0.8185**, representing a **57.7% error reduction** over the naive baseline.

---

### Ames Housing Results

#### 1. 5-Fold Cross-Validation Performance (Training Set)
*Models sorted by mean Cross-Validation RMSE (lowest to highest).*

| Rank | Model | CV MAE (Mean) | CV MAE (Std) | CV RMSE (Mean) | CV RMSE (Std) | CV $R^2$ (Mean) | CV $R^2$ (Std) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost** | 14,915.10 | 1,393.41 | 24,277.95 | 5,450.49 | 0.90 | 0.05 |
| 2 | **Gradient Boosting** | 15,095.39 | 1,503.63 | 24,717.69 | 6,328.71 | 0.89 | 0.06 |
| 3 | **Random Forest** | 16,080.99 | 1,484.47 | 26,596.70 | 5,812.55 | 0.88 | 0.05 |
| 4 | **Lasso Regression** | 16,351.75 | 1,337.09 | 27,951.15 | 9,163.32 | 0.85 | 0.10 |
| 5 | **Linear Regression** | 16,425.69 | 1,355.66 | 28,193.65 | 9,482.06 | 0.85 | 0.11 |
| 6 | **Elastic Net** | 16,571.96 | 1,124.97 | 29,034.87 | 8,975.94 | 0.84 | 0.10 |
| 7 | **Ridge Regression** | 16,578.45 | 1,118.58 | 29,053.63 | 8,973.10 | 0.84 | 0.10 |
| 8 | **Decision Tree** | 24,353.90 | 1,403.42 | 37,968.25 | 4,680.88 | 0.75 | 0.07 |

#### 2. Final Test Set Performance
*Sorted by Test RMSE (lowest to highest). Naive baseline Mean RMSE is 90,222.37.*

| Model | Test MAE | Test RMSE | Test $R^2$ | MAPE (%) | Train MAE | Train RMSE | Train $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **15,045.41** | **25,100.92** | **0.9214** | **8.04** | 10,670.28 | 14,818.71 | 0.9631 |
| Gradient Boosting | 15,165.20 | 25,928.42 | 0.9161 | 8.06 | 11,235.03 | 15,312.88 | 0.9606 |
| Random Forest | 15,729.69 | 26,827.57 | 0.9102 | 8.53 | 6,510.45 | 11,962.47 | 0.9759 |
| Tuned XGBoost | 15,757.57 | 26,925.08 | 0.9096 | 8.51 | 6,101.71 | 9,993.93 | 0.9832 |
| Ridge Regression | 16,186.78 | 28,896.36 | 0.8959 | 8.77 | 13,737.70 | 20,470.24 | 0.9295 |
| Elastic Net | 16,243.90 | 28,923.28 | 0.8957 | 8.78 | 13,794.94 | 20,635.38 | 0.9284 |
| Lasso Regression | 15,868.93 | 29,253.40 | 0.8933 | 9.18 | 13,141.08 | 18,986.06 | 0.9394 |
| Linear Regression | 16,007.38 | 29,615.38 | 0.8906 | 9.28 | 13,125.11 | 18,980.98 | 0.9394 |
| Decision Tree | 22,329.61 | 32,284.94 | 0.8700 | 12.59 | 17,984.22 | 25,324.26 | 0.8921 |

> [!NOTE]
> **Best Ames Model**: **XGBoost** (default params: `learning_rate=0.3`, `max_depth=6`, `n_estimators=100`) achieved the lowest Test RMSE of **25,100.92** and an $R^2$ of **0.9214**. This reduces error by **72.2%** over the naive baseline. 
> The Tuned XGBoost (`learning_rate=0.03`, `max_depth=4`, `n_estimators=300`, `subsample=0.8`) achieved a slightly lower test performance on this particular split (RMSE: 26,925.08, $R^2$: 0.9096), although it yielded a better average cross-validation score on the train set (CV RMSE: 23,327.46 vs 24,277.95).

---

## Reproducing the Results

To set up the project environment and run the notebooks:

### Prerequisites
* Python 3.10.x (verified on `3.10.0`)
* Virtualenv or Conda package manager

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd House_price_pred
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install numpy==2.2.6 pandas==2.3.2 scikit-learn==1.7.2 matplotlib==3.10.6 seaborn==0.13.2 xgboost==2.1.1 joblib==1.4.2 jupyter
   # Note: you may need 'ucimlrepo' to fetch the UCI dataset
   pip install ucimlrepo
   ```

4. **Run the Notebooks**:
   ```bash
   jupyter notebook
   ```
   Open and run all cells in [uci_house.ipynb](file:///c:/Users/apara/Downloads/VIT%20Downloads/Predictive%20lab/House_price_pred/uci_house.ipynb) and [ames_housing.ipynb](file:///c:/Users/apara/Downloads/VIT%20Downloads/Predictive%20lab/House_price_pred/ames_housing.ipynb).

---

## Serialized Model Artifacts

Trained pipelines and estimators are saved in the `artifacts/` folder as `.joblib` files. These files package the complete scikit-learn pipeline (both preprocessing steps and estimator), allowing easy deserialization and deployment for inference.

### How to load a model and make predictions:

```python
import joblib
import pandas as pd

# 1. Load the best pipeline (contains both ColumnTransformer and Estimator)
model_pipeline = joblib.load('artifacts/uci_best_model.joblib')

# 2. Prepare raw data (matching the original dataset format before scaling/encoding)
new_data = pd.DataFrame([{
    'X1 transaction date': 2013.500,
    'X2 house age': 13.3,
    'X3 distance to the nearest MRT station': 561.9845,
    'X4 number of convenience stores': 5,
    'X5 latitude': 24.98746,
    'X6 longitude': 121.54391
}])

# 3. Predict directly (preprocessing will be applied automatically by the pipeline)
predicted_price = model_pipeline.predict(new_data)
print(f"Predicted Price (unit area): {predicted_price[0]:.2f} (10,000 NTD/Ping)")
```

---

## CLI Tool for Inference & Retraining

The `house_price_cli.py` script provides a command-line interface for loading trained models, making predictions on new data, retraining models using saved preprocessing pipelines, and managing model artifacts. It supports both the **UCI Real Estate Valuation** and **Ames Housing** datasets.

### Prerequisites

Ensure you have activated the virtual environment and installed dependencies as described in [Reproducing the Results](#reproducing-the-results). The notebooks must be run at least once to generate the serialized model artifacts in the `artifacts/` directory.

### Commands Overview

| Command | Description |
|---------|-------------|
| `predict` | Load the best model and make predictions on new data (JSON string, JSON file, or CSV) |
| `retrain` | Load the saved preprocessing pipeline and retrain all models on a new dataset |
| `list-models` | Display all available models, their performance metrics, and artifact files |
| `interactive` | Interactive prompt to enter feature values manually and get predictions |

```
usage: house_price_cli.py [-h] {predict,retrain,list-models,interactive} ...

House Price Prediction CLI - Load models, predict, retrain, and manage
artifacts for UCI and Ames datasets.

positional arguments:
  {predict,retrain,list-models,interactive}
                        Available commands

optional arguments:
  -h, --help            show this help message and exit
```

---

### Prediction Examples

#### 1. Predict using a JSON string (UCI dataset)

```bash
python house_price_cli.py predict --dataset uci --input '{
    "X1 transaction date": 2013.500,
    "X2 house age": 13.3,
    "X3 distance to the nearest MRT station": 561.9845,
    "X4 number of convenience stores": 5,
    "X5 latitude": 24.98746,
    "X6 longitude": 121.54391
}'
```

#### 2. Predict multiple records from a JSON file

Create `sample_uci.json`:
```json
[
    {
        "X1 transaction date": 2013.500,
        "X2 house age": 13.3,
        "X3 distance to the nearest MRT station": 561.9845,
        "X4 number of convenience stores": 5,
        "X5 latitude": 24.98746,
        "X6 longitude": 121.54391
    },
    {
        "X1 transaction date": 2012.917,
        "X2 house age": 32.0,
        "X3 distance to the nearest MRT station": 217.5,
        "X4 number of convenience stores": 4,
        "X5 latitude": 24.98298,
        "X6 longitude": 121.54024
    }
]
```

```bash
python house_price_cli.py predict --dataset uci --input sample_uci.json
```

#### 3. Predict on Ames dataset using a CSV file

```bash
python house_price_cli.py predict --dataset ames --input new_houses.csv --output predictions.csv
```

The input CSV should contain the same feature columns used during training (all columns except `SalePrice`). If `SalePrice` is present, it will be ignored for prediction.

#### 4. Save predictions to an output file

```bash
python house_price_cli.py predict --dataset uci --input sample_uci.json --output result.json
```

---

### Retraining Examples

The `retrain` command loads the saved **preprocessing pipeline** (fitted `ColumnTransformer`) from the original notebook training, then retrains all regression models on a new dataset. This ensures consistent feature transformations while updating model weights with fresh data.

#### 1. Retrain UCI models on new data

```bash
python house_price_cli.py retrain --dataset uci --data new_uci_data.csv
```

The CSV must contain the 6 feature columns (`X1 transaction date`, `X2 house age`, `X3 distance to the nearest MRT station`, `X4 number of convenience stores`, `X5 latitude`, `X6 longitude`) and the target column `Y house price of unit area`.

#### 2. Retrain Ames models on new data

```bash
python house_price_cli.py retrain --dataset ames --data new_ames_data.csv
```

The CSV must contain the 79 feature columns and the `SalePrice` target column.

**What happens during retraining:**

1. **Preprocessor loaded** — The saved `ColumnTransformer` (with fitted imputation values, scaling parameters, and one-hot encoding categories) is loaded from the artifacts directory.
2. **New data loaded** — The training CSV is read and split into features (`X`) and target (`y`).
3. **Models retrained** — 8 regressors (Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost) are trained using the loaded preprocessor.
4. **Best model selected** — Models are evaluated on the training data; the one with the lowest RMSE is saved as the new best model.
5. **Artifacts updated** — The best model, all individual models, a results CSV, and updated metadata are saved to the `artifacts/` directory, overwriting previous versions.

---

### Listing Models

Display all trained models, their performance metrics, best model info, tuned configurations (UCI only), and available artifact files:

```bash
# List UCI models
python house_price_cli.py list-models --dataset uci

# List Ames models
python house_price_cli.py list-models --dataset ames
```

Example output for UCI:
```
============================================================
  MODEL REGISTRY: UCI
============================================================

  All Models Performance (sorted by RMSE):
  ------------------------------------------------------------
      Model    RMSE      MAE       R2
  Tuned Random Forest    5.5174    3.7342    0.8185
  XGBoost               5.5327    3.7615    0.8175
  ...

  Best Model Info:
  ----------------------------------------
    Name:    Tuned Random Forest
    File:    best_model.joblib
    RMSE:    5.5174
    R2:      0.8185
    MAE:     3.7342
    MAPE:    11.6080

  Tuned Model Configurations:
  ----------------------------------------
    Random_Forest:
      model__max_depth: 10
      model__min_samples_leaf: 2
      model__n_estimators: 200
    ...

  Available Artifact Files:
  ----------------------------------------
    uci_Tuned_Random_Forest.joblib (195.3 KB)
    uci_best_model.joblib (195.3 KB)
    ...
```

---

### Interactive Mode

Launch an interactive prompt to enter feature values manually and receive predictions:

```bash
# Interactive mode for UCI (6 features)
python house_price_cli.py interactive --dataset uci

# Interactive mode for Ames (prompts for CSV or uses sample)
python house_price_cli.py interactive --dataset ames
```

For the **UCI dataset**, the tool will prompt for each of the 6 features individually with descriptions. For the **Ames dataset**, you can provide a CSV file path, or the tool will automatically use a sample record from `AmesHousing.csv` for demonstration.

```
============================================================
  INTERACTIVE MODE: UCI
============================================================

  Enter feature values. Type 'quit' to exit.

  --------------------------------------------------
  New Prediction
  --------------------------------------------------
    X1 transaction date (Transaction date (e.g., 2013.250 = 2013 March)): 2013.5
    X2 house age (Age of the house in years): 13.3
    X3 distance to the nearest MRT station (Distance to nearest MRT station (meters)): 561.98
    X4 number of convenience stores (Number of convenience stores in walking distance): 5
    X5 latitude (Latitude coordinate (degrees)): 24.987
    X6 longitude (Longitude coordinate (degrees)): 121.544

    >>> Predicted Y house price of unit area: 46.7200 10,000 NTD/Ping

  --------------------------------------------------
  New Prediction
  ...
```

---

## Environment & Dependencies

Training was run and verified under the following package versions:
* **Python**: 3.10.0
* **NumPy**: 2.2.6
* **Pandas**: 2.3.2
* **Scikit-learn**: 1.7.2
* **XGBoost**: 2.1.1
* **Joblib**: 1.4.2
* **Matplotlib**: 3.10.6
* **Seaborn**: 0.13.2
* **UciMLRepo**: 0.0.7
