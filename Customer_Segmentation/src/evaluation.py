"""
Evaluation and Visualization Module for Customer Segmentation Models.
Provides standard metrics computation (Accuracy, Precision, Recall, F1, ROC-AUC)
and visualization routines for model comparison, confusion matrices, and ROC curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def compute_model_metrics(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Computes comprehensive evaluation metrics for a fitted classifier model pipeline.
    """
    y_pred = model.predict(X_test)
    
    # Try predicting probabilities for ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)
        # Multiclass ROC-AUC using OvR strategy
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except (AttributeError, NotImplementedError):
        roc_auc = np.nan
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision (Weighted)': prec_weighted,
        'Recall (Weighted)': rec_weighted,
        'F1-Score (Weighted)': f1_weighted,
        'F1-Score (Macro)': f1_macro,
        'ROC-AUC (Weighted OvR)': roc_auc
    }
    return metrics

def build_comparison_dataframe(results_list: list) -> pd.DataFrame:
    """
    Converts a list of model metrics dictionaries into a sorted summary DataFrame.
    """
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    return df_results

def plot_model_comparison(df_results: pd.DataFrame, save_path: str = None):
    """
    Plots a dual bar chart comparing Accuracy and F1-Score across evaluated models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df_results.melt(id_vars=['Model'], value_vars=['Accuracy', 'F1-Score (Weighted)'],
                              var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_plot, x='Score', y='Model', hue='Metric', palette='viridis', ax=ax)
    ax.set_title('Model Performance Comparison (Accuracy & F1-Score)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.3f}',
                        (width + 0.01, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9)
                        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_confusion_matrices(models_dict: dict, X_test, y_test, class_names=None, save_path: str = None):
    """
    Plots heatmaps of confusion matrices for a dictionary of fitted models.
    """
    n_models = len(models_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names if class_names else 'auto',
                    yticklabels=class_names if class_names else 'auto')
        ax.set_title(f'Confusion Matrix: {name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Segment')
        ax.set_ylabel('Actual Segment')
        
    # Hide extra axes if any
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_multiclass_roc(model, X_test, y_test, model_name: str, class_names: list, save_path: str = None):
    """
    Plots One-vs-Rest ROC curves for a multiclass classification model.
    """
    y_proba = model.predict_proba(X_test)
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette('Set1', n_classes)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'Class {class_names[i]} (AUC = {roc_auc:.3f})')
                
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance / Baseline')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title(f'Multiclass ROC Curves - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_pca_clusters(df_segmented: pd.DataFrame, save_path: str = None):
    """
    Plots PCA 2D scatter of discovered customer segments.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df_segmented, x='PCA1', y='PCA2', hue='Segment_Label',
        style='Segment_Label', palette='tab10', s=70, alpha=0.8, ax=ax
    )
    ax.set_title('Customer Segments in 2D PCA Space', fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component 1', fontsize=11)
    ax.set_ylabel('Principal Component 2', fontsize=11)
    ax.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
