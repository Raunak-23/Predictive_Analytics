"""
Model Construction, Pipeline Assembly, and Hyperparameter Grid Definitions.
Supports multiple Naive Bayes algorithms and benchmark classifiers for customer segmentation.
"""

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import (
    create_standard_preprocessor,
    create_nonnegative_preprocessor,
    create_categorical_nb_preprocessor
)

def build_model_pipelines(random_state: int = 42) -> dict:
    """
    Constructs end-to-end Scikit-Learn pipelines for Naive Bayes variants and comparison classifiers.
    """
    std_preprocessor = create_standard_preprocessor()
    nonneg_preprocessor = create_nonnegative_preprocessor()
    cat_nb_preprocessor = create_categorical_nb_preprocessor(n_bins=5)

    pipelines = {
        'Gaussian Naive Bayes': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', GaussianNB())
        ]),
        'Categorical Naive Bayes': Pipeline(steps=[
            ('preprocessor', cat_nb_preprocessor),
            ('classifier', CategoricalNB(min_categories=6))
        ]),
        'Bernoulli Naive Bayes': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', BernoulliNB())
        ]),
        'Complement Naive Bayes': Pipeline(steps=[
            ('preprocessor', nonneg_preprocessor),
            ('classifier', ComplementNB())
        ]),
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=random_state))
        ]),
        'Support Vector Machine': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', SVC(probability=True, random_state=random_state))
        ]),
        'K-Nearest Neighbors': Pipeline(steps=[
            ('preprocessor', std_preprocessor),
            ('classifier', KNeighborsClassifier())
        ])
    }
    
    return pipelines

def get_hyperparameter_grids() -> dict:
    """
    Returns hyperparameter search grids for GridSearchCV tuning across models.
    """
    param_grids = {
        'Gaussian Naive Bayes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },
        'Categorical Naive Bayes': {
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        },
        'Bernoulli Naive Bayes': {
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'classifier__binarize': [0.0, 0.1, 0.5, None]
        },
        'Complement Naive Bayes': {
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'classifier__norm': [False, True]
        },
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs', 'saga']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5]
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 3, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        },
        'Support Vector Machine': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear']
        },
        'K-Nearest Neighbors': {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance']
        }
    }
    return param_grids
