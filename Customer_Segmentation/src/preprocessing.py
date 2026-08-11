"""
Data Preprocessing and Scikit-Learn Pipeline Transformers for Customer Segmentation.
Provides modular ColumnTransformers tailored for continuous, categorical, and mixed Naive Bayes models.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Define feature groups based on domain analysis
NUMERICAL_FEATURES = ['Age', 'Income']
CATEGORICAL_FEATURES = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def create_standard_preprocessor() -> ColumnTransformer:
    """
    Creates a standard ColumnTransformer combining:
    - StandardScaler for numerical continuous features ('Age', 'Income')
    - OneHotEncoder for categorical features ('Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size')
    Ideal for GaussianNB, Logistic Regression, Random Forest, SVM, KNN.
    """
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def create_nonnegative_preprocessor() -> ColumnTransformer:
    """
    Creates a MinMaxScaler + OneHotEncoder preprocessor producing non-negative feature matrices.
    Ideal for ComplementNB and MultinomialNB models.
    """
    num_transformer = Pipeline(steps=[
        ('minmax', MinMaxScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def create_categorical_nb_preprocessor(n_bins: int = 5) -> ColumnTransformer:
    """
    Creates an ordinal integer preprocessor for CategoricalNB.
    Discretizes numerical continuous features into discrete bins and ordinal encodes categorical features.
    """
    num_discretizer = Pipeline(steps=[
        ('binning', KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile'))
    ])
    
    cat_encoder = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_discretizer, NUMERICAL_FEATURES),
            ('cat', cat_encoder, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def split_customer_data(df: pd.DataFrame, target_col: str = 'Segment', test_size: float = 0.2, random_state: int = 42):
    """
    Splits dataset into stratified train and test sets.
    """
    X = df[ALL_FEATURES]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
