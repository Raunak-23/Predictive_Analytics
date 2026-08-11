"""
Utility Module for Artifact Management, Serialization, and Production Inference.
Handles joblib saving/loading of model pipelines and JSON export of metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

def save_pipeline_artifact(pipeline_obj, filepath: str):
    """
    Saves a Scikit-Learn pipeline or model object to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline_obj, filepath)
    print(f"Artifact successfully saved to: {filepath}")

def load_pipeline_artifact(filepath: str):
    """
    Loads a saved pipeline or model artifact from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact file not found at: {filepath}")
    model_obj = joblib.load(filepath)
    return model_obj

def save_metrics_json(metrics_data: list, filepath: str):
    """
    Saves evaluation metrics list to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Handle NaN values for JSON serialization
    cleaned_metrics = []
    for item in metrics_data:
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                clean_item[k] = None
            else:
                clean_item[k] = v
        cleaned_metrics.append(clean_item)
        
    with open(filepath, 'w') as f:
        json.dump(cleaned_metrics, f, indent=4)
    print(f"Metrics JSON saved to: {filepath}")

def predict_single_customer(pipeline_obj, customer_dict: dict) -> dict:
    """
    Runs production inference for a new customer given a dict of feature inputs.
    """
    df_single = pd.DataFrame([customer_dict])
    pred_segment = pipeline_obj.predict(df_single)[0]
    
    segment_names = {
        0: "Fewer Opportunities",
        1: "Standard",
        2: "Career-Focused",
        3: "Well-Off"
    }
    
    try:
        probabilities = pipeline_obj.predict_proba(df_single)[0]
        prob_dict = {segment_names.get(i, f"Segment_{i}"): float(p) for i, p in enumerate(probabilities)}
    except (AttributeError, NotImplementedError):
        prob_dict = None
        
    return {
        'Predicted_Segment_ID': int(pred_segment),
        'Predicted_Segment_Name': segment_names.get(int(pred_segment), f"Segment_{pred_segment}"),
        'Class_Probabilities': prob_dict
    }
