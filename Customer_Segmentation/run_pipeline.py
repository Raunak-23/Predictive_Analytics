"""
End-to-End Pipeline Execution Script for Customer Segmentation using Naive Bayes.
Executes all 7 stages, saves plot figures, exports model artifacts, and verifies production inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Add current directory to path
sys.path.append(os.getcwd())

from src.data import load_raw_data, discover_customer_segments
from src.preprocessing import split_customer_data
from src.models import build_model_pipelines, get_hyperparameter_grids
from src.evaluation import (
    compute_model_metrics, build_comparison_dataframe,
    plot_model_comparison, plot_confusion_matrices,
    plot_multiclass_roc, plot_pca_clusters
)
from src.utils import (
    save_pipeline_artifact, load_pipeline_artifact,
    save_metrics_json, predict_single_customer
)

def run_full_pipeline():
    print("=" * 70)
    print("STARTING 7-STAGE CUSTOMER SEGMENTATION PIPELINE (NAIVE BAYES FOCUS)")
    print("=" * 70)
    
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('artifacts/plots', exist_ok=True)
    
    # ------------------------------------------------------------------------
    # STAGE 1: Data and Library Import
    # ------------------------------------------------------------------------
    print("\n---> STAGE 1: Data & Library Import")
    csv_path = 'segmentation data.csv'
    df_raw = load_raw_data(csv_path)
    print(f"Loaded raw dataset shape: {df_raw.shape}")
    print(df_raw.head())
    
    # ------------------------------------------------------------------------
    # STAGE 2: Exploratory Data Analysis (EDA) & Segment Discovery
    # ------------------------------------------------------------------------
    print("\n---> STAGE 2: Exploratory Data Analysis & Segment Discovery")
    df_segmented, kmeans_model, scaler_model, pca_model = discover_customer_segments(df_raw, n_clusters=4)
    print("Customer Segments Discovered:")
    print(df_segmented['Segment_Label'].value_counts())
    
    fig_pca = plot_pca_clusters(df_segmented, save_path='artifacts/plots/pca_clusters.png')
    plt.close(fig_pca)
    
    # Save segmentation baseline artifacts
    save_pipeline_artifact(kmeans_model, 'artifacts/kmeans_segmenter.joblib')
    save_pipeline_artifact(scaler_model, 'artifacts/kmeans_scaler.joblib')
    
    # ------------------------------------------------------------------------
    # STAGE 3: Data Preprocessing & Train/Test Split
    # ------------------------------------------------------------------------
    print("\n---> STAGE 3: Data Preprocessing & Split")
    X_train, X_test, y_train, y_test = split_customer_data(df_segmented, target_col='Segment', test_size=0.2)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # ------------------------------------------------------------------------
    # STAGE 4: Model Building (Pipeline Assembly)
    # ------------------------------------------------------------------------
    print("\n---> STAGE 4: Model Building")
    raw_pipelines = build_model_pipelines(random_state=42)
    print(f"Instantiated {len(raw_pipelines)} model pipelines:")
    for name in raw_pipelines:
        print(f"  - {name}")
        
    # ------------------------------------------------------------------------
    # STAGE 5: Hyperparameter Tuning (GridSearchCV)
    # ------------------------------------------------------------------------
    print("\n---> STAGE 5: Hyperparameter Tuning (GridSearchCV)")
    param_grids = get_hyperparameter_grids()
    
    tuned_pipelines = {}
    best_params_summary = {}
    
    for name, pipeline in raw_pipelines.items():
        print(f"Tuning {name}...")
        grid_params = param_grids.get(name, {})
        if grid_params:
            search = GridSearchCV(
                pipeline, param_grid=grid_params, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            search.fit(X_train, y_train)
            tuned_pipelines[name] = search.best_estimator_
            best_params_summary[name] = search.best_params_
            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV F1-Score: {search.best_score_:.4f}")
        else:
            pipeline.fit(X_train, y_train)
            tuned_pipelines[name] = pipeline
            best_params_summary[name] = "Default Parameters"
            
    # ------------------------------------------------------------------------
    # STAGE 6: Model Evaluation and Comparison
    # ------------------------------------------------------------------------
    print("\n---> STAGE 6: Model Evaluation and Comparison")
    evaluation_results = []
    
    for name, model in tuned_pipelines.items():
        metrics = compute_model_metrics(model, X_test, y_test, model_name=name)
        evaluation_results.append(metrics)
        
    df_comparison = build_comparison_dataframe(evaluation_results)
    print("\nModel Comparison Table:")
    print(df_comparison.to_string(index=False))
    
    # ------------------------------------------------------------------------
    # STAGE 7: Visualizations & Artifact Saving
    # ------------------------------------------------------------------------
    print("\n---> STAGE 7: Visualizations & Artifact Saving")
    
    # Plot comparison bar chart
    fig_comp = plot_model_comparison(df_comparison, save_path='artifacts/plots/model_comparison.png')
    plt.close(fig_comp)
    
    # Plot confusion matrices for key Naive Bayes and Top models
    selected_models = {
        'Gaussian NB': tuned_pipelines['Gaussian Naive Bayes'],
        'Categorical NB': tuned_pipelines['Categorical Naive Bayes'],
        'Bernoulli NB': tuned_pipelines['Bernoulli Naive Bayes'],
        'Complement NB': tuned_pipelines['Complement Naive Bayes'],
        'Random Forest': tuned_pipelines['Random Forest'],
        'Logistic Regression': tuned_pipelines['Logistic Regression']
    }
    class_labels = ["Fewer Opp.", "Standard", "Career-Foc.", "Well-Off"]
    fig_cm = plot_confusion_matrices(selected_models, X_test, y_test, class_names=class_labels,
                                     save_path='artifacts/plots/confusion_matrices.png')
    plt.close(fig_cm)
    
    # Plot ROC curve for Gaussian Naive Bayes
    gnb_model = tuned_pipelines['Gaussian Naive Bayes']
    fig_roc = plot_multiclass_roc(gnb_model, X_test, y_test, model_name='Gaussian Naive Bayes',
                                  class_names=class_labels, save_path='artifacts/plots/gaussian_nb_roc.png')
    plt.close(fig_roc)
    
    # Save trained pipelines
    save_pipeline_artifact(gnb_model, 'artifacts/gaussian_nb_pipeline.joblib')
    save_pipeline_artifact(tuned_pipelines['Categorical Naive Bayes'], 'artifacts/categorical_nb_pipeline.joblib')
    
    best_model_name = df_comparison.iloc[0]['Model']
    best_model_pipeline = tuned_pipelines[best_model_name]
    save_pipeline_artifact(best_model_pipeline, 'artifacts/best_model_pipeline.joblib')
    print(f"\nSaved best model pipeline ({best_model_name}) to artifacts/best_model_pipeline.joblib")
    
    # Export metrics JSON
    save_metrics_json(evaluation_results, 'artifacts/model_metrics.json')
    
    # Production Inference Verification
    print("\n---> Verifying Saved Pipeline Reload & Sample Customer Inference:")
    reloaded_gnb = load_pipeline_artifact('artifacts/gaussian_nb_pipeline.joblib')
    sample_customer = {
        'Sex': 0,
        'Marital status': 0,
        'Age': 45,
        'Education': 2,
        'Income': 175000,
        'Occupation': 2,
        'Settlement size': 2
    }
    pred_res = predict_single_customer(reloaded_gnb, sample_customer)
    print(f"Sample Customer Input: {sample_customer}")
    print(f"Reloaded Pipeline Prediction: {pred_res['Predicted_Segment_Name']} (ID: {pred_res['Predicted_Segment_ID']})")
    print(f"Class Probabilities: {pred_res['Class_Probabilities']}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY WITH ALL ARTIFACTS GENERATED!")
    print("=" * 70)

if __name__ == '__main__':
    run_full_pipeline()
