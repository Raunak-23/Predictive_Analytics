"""
Data Loading and Segmentation Module for Customer Segmentation project.
Handles raw dataset loading, validation, and ground-truth segment label generation using K-Means clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw customer segmentation CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def discover_customer_segments(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> tuple:
    """
    Perform baseline unsupervised K-Means clustering to discover customer segments
    and return dataframe with 'Segment' target column, cluster model, and fitted scaler.
    """
    feature_cols = [c for c in df.columns if c not in ['ID', 'Segment', 'PCA1', 'PCA2', 'Segment_Label']]
    X = df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    
    df_segmented = df.copy()
    df_segmented['Segment'] = cluster_labels
    df_segmented['PCA1'] = pca_coords[:, 0]
    df_segmented['PCA2'] = pca_coords[:, 1]
    
    # Define Segment names based on domain characteristics
    segment_map = {
        0: "Fewer Opportunities",
        1: "Standard",
        2: "Career-Focused",
        3: "Well-Off"
    }
    df_segmented['Segment_Label'] = df_segmented['Segment'].map(segment_map)
    
    return df_segmented, kmeans, scaler, pca
