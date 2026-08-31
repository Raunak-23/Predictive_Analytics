import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Any
from checkpointing import run_resumable
from evaluation import evaluate

def rolling_origins(series, initial, horizon, step):
    """
    Generate (train, test) slices for rolling-origin evaluation.
    Yields (train_series, test_series) for each fold.
    Stops when the next fold would exceed the series length.
    """
    end = initial
    while end + horizon <= len(series):
        yield series.iloc[:end], series.iloc[end:end+horizon]
        end += step

def rolling_origin_cv(pipeline_class, pipeline_kwargs, series,
                      initial, horizon, step, results_path,
                      desc="Rolling-origin CV"):
    """
    Perform rolling-origin cross-validation with checkpointing.

    Parameters
    ----------
    pipeline_class : class
        A subclass of BaseForecastPipeline.
    pipeline_kwargs : dict
        Keyword arguments for instantiating the pipeline.
    series : pd.Series
        The full time series.
    initial : int
        Number of initial periods for the first training set.
    horizon : int
        Forecast horizon for each fold.
    step : int
        Step size between origins.
    results_path : str or Path
        Path to a .jsonl file where fold results will be appended.
    desc : str
        Description for the progress bar.

    Returns
    -------
    pd.DataFrame
        Table of fold results (including fold index, MAE, RMSE).
    """
    results_path = Path(results_path)

    # Build the list of folds; each item is a tuple (fold_idx, train_slice, test_slice)
    folds = []
    for idx, (train_slice, test_slice) in enumerate(rolling_origins(series, initial, horizon, step)):
        folds.append((idx, train_slice, test_slice))

    # Define key function: fold index + model name for uniqueness
    model_name = pipeline_class.__name__
    def key_fn(fold_item):
        idx, _, _ = fold_item
        return f"{model_name}_fold_{idx}"

    # Define process function: fit a fresh pipeline on training slice, predict, score
    def process_fn(fold_item):
        idx, train_slice, test_slice = fold_item
        # Instantiate a fresh pipeline
        pipeline = pipeline_class(**pipeline_kwargs)
        pipeline.fit(train_slice)
        pred = pipeline.predict(len(test_slice))
        scores = evaluate(test_slice, pred)
        return {
            'fold': idx,
            'train_start': train_slice.index[0].strftime('%Y-%m-%d'),
            'train_end': train_slice.index[-1].strftime('%Y-%m-%d'),
            'test_start': test_slice.index[0].strftime('%Y-%m-%d'),
            'test_end': test_slice.index[-1].strftime('%Y-%m-%d'),
            'MAE': scores['MAE'],
            'RMSE': scores['RMSE']
        }

    # Run resumably
    run_resumable(
        items=folds,
        key_fn=key_fn,
        process_fn=process_fn,
        results_path=results_path,
        desc=desc,
        key_field='key'
    )

    # Read all completed results back into a DataFrame.
    # run_resumable stores records FLAT: the process_fn dict has 'key' injected
    # at the top level (not nested under a 'result' sub-key), so we read fields
    # directly from each parsed JSON object.
    completed = []
    if results_path.exists():
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Each record is flat: {'fold': int, 'MAE': float, ..., 'key': str}
                    if 'fold' in data and 'MAE' in data and 'RMSE' in data:
                        completed.append({
                            'fold':       data['fold'],
                            'train_start': data.get('train_start'),
                            'train_end':   data.get('train_end'),
                            'test_start':  data.get('test_start'),
                            'test_end':    data.get('test_end'),
                            'MAE':         data['MAE'],
                            'RMSE':        data['RMSE'],
                        })
                except Exception:
                    continue
    if not completed:
        print("No completed folds found in checkpoint file.")
        return pd.DataFrame()
    df = pd.DataFrame(completed).sort_values('fold').reset_index(drop=True)
    return df
