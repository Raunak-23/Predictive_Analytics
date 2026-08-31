# MDI3003 Lab 06 — Spatio-Temporal Incident Forecasting & Autoregressive Time-Series Modeling

> **Course**: MDI3003 Advanced Predictive Analytics / Predictive Analytics Lab (Lab 06)  
> **Institution**: Vellore Institute of Technology (VIT), Vellore  
> **Author**: Raunak Pal (Reg. No: 23MID0045)  
> **Topic**: Leakage-Safe Spatio-Temporal Forecasting of Weekly Crime Incident Counts via AR, ARIMA, SARIMA, and Deep Learning Architectures  
> **Dataset**: City of Chicago Reported Crime Incidents (2001 to Present — 8,623,069 Records)  
> **Evaluation Protocol**: 12-Week Chronological Out-of-Sample Holdout + 83-Fold Checkpointed Rolling-Origin Walk-Forward Backtesting  

---

## 1. Executive Summary & Problem Framing

This repository provides an end-to-end, leakage-safe, production-ready time-series forecasting framework engineered for **MDI3003 Lab 06**. The study formulates incident forecasting as a spatial-temporal regression problem over continuous aggregate weekly counts. It benchmarks classical statistical baselines, autoregressive formulations, box-jenkins ARIMA/SARIMA specifications, and recurrent neural networks across municipal police jurisdictions in Chicago, Illinois.

```
+----------------------------------------------------------------------------------------------------+
|                                    TIME-SERIES FORECASTING HIERARCHY                               |
+------------------------------------+-----------------------------------+---------------------------+
| Baseline: Naive (Persistence)      | Statistical: AR(3) & ARIMA(2,1,0) | Advanced: SARIMA & LSTM   |
| $Y_{T+h} = Y_T$                    | Checkpointed Grid Search via AIC  | 52-Wk Seasonal & PyTorch  |
| Locked Holdout MAE: 53.58          | Locked Holdout MAE: 26.99 / 46.13 | Locked Holdout MAE: 34.51 |
+------------------------------------+-----------------------------------+---------------------------+
| 83-Fold Walk-Forward Backtest      | Multi-District Spatial Scaling    | Ethical Boundary Protocol |
| ARIMA Mean MAE: 27.87 ± 12.74      | 21 Police Districts Replicated    | Strictly Aggregate Counts |
| SARIMA Mean MAE: 29.16 ± 11.87     | Batch cuML GPU / CPU Fallback     | Zero Person-Level Profiling|
+------------------------------------+-----------------------------------+---------------------------+
```

### 1.1 Core Objectives & Theoretical Motivation
1. **Spatio-Temporal Aggregation**: Ingest and structure 8.62M raw incident records spanning January 1, 2001 through August 24, 2026 into continuous, equidistant weekly time series (`W-MON`) across municipal police districts.
2. **Spatial Feature Separation Principle**: Police districts represent heterogeneous socio-demographic and geographic partitions. Encoding districts as numeric features in a single linear model introduces spurious metric distance and ordering. The system models each jurisdiction as an independent, univariate time series.
3. **Leakage-Safe Chronological Validation**: Strictly enforce temporal quarantine by splitting the 1,339 continuous weekly observations into an in-sample training partition (1,327 weeks: 2001-01-01 to 2026-06-01) and a locked out-of-sample forward test horizon (12 weeks: 2026-06-08 to 2026-08-24).
4. **Universal Walk-Forward Cross-Validation**: Address the fundamental limitation of single-holdout evaluations via an 83-fold rolling-origin backtesting engine (initial training window = 668 weeks / ~13 years, step = 8 weeks, horizon = 8 weeks).
5. **Responsible-Use & Governance Assurance**: Formalize operational boundaries differentiating administrative volume forecasting from punitive predictive policing, establishing protocols that prevent algorithmic feedback loops.

### 1.2 Key Empirical Findings

| Model Architecture | Parameter Order / Config | 12-Wk Test MAE | 12-Wk Test RMSE | 83-Fold RO Mean MAE | 83-Fold RO Std MAE | 83-Fold RO Mean RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive (Persistence)** | $Y_{T+h} = Y_T$ (Baseline) | 53.58 | 76.65 | — | — | — |
| **Autoregressive (AR)** | $p=3$, trend=`'ct'` | **26.99** | **49.81** | 30.14 | 13.78 | 35.32 |
| **ARIMA (Selected)** | $(p=2, d=1, q=0)$, trend=`'n'` | 46.13 | 71.25 | **27.87** | **12.74** | **32.38** |
| **SARIMA (Seasonal)** | $(2, 1, 0) \times (1, 1, 1)_{52}$ | 34.51 | 63.73 | 29.16 | 11.87 | 34.90 |
| **LSTM Neural Network** | 2-Layer PyTorch, Win=12 | 36.08 | 60.21 | — | — | — |

* **The "Holdout Paradox" Resolved**: On the single 12-week test holdout, AR(3) posted the lowest point error (MAE 26.99 vs ARIMA 46.13). However, across 83 rolling-origin backtesting folds, **ARIMA(2, 1, 0)** demonstrated superior temporal stability (Mean MAE **27.87 ± 12.74** vs AR(3) **30.14 ± 13.78**). ARIMA is the empirically validated champion for continuous deployment.
* **Seasonal Integration**: Incorporating an annual 52-week seasonal differencing term ($s=52$) via SARIMA slashed test MAE from 46.13 down to **34.51**, accounting for summer peaks and winter troughs.
* **Spatial Replication (District 011 Harrison)**: Replicating the pipeline on Chicago's second highest-volume district revealed non-identical stochastic dynamics ($d=0$ stationarity vs. $d=1$ for District 008), proving that time-series models must be independently fitted per spatial unit rather than transferred blindly.

---

## 2. Repository Structure & Artifact Ecosystem

The repository is structured as a self-contained, reproducible, modular analytics pipeline. All outputs, checkpointed logs, models, figures, and manifests are versioned.

```text
Crime_Indicent_Analysis/
├── README.md                                                 # Master technical documentation (this file)
├── requirements.txt                                          # Pinned environment & framework dependencies
│
├── notebooks/                                                # Full interactive research & experimental notebooks
│   ├── chicago_crimes.ipynb                                  # Master 159-cell analysis notebook (Stages 0–14)
│   ├── chicago_crimes.html                                   # Compiled HTML report export with all figures & logs
│   └── chicago_crimes.pdf                                    # Formatted PDF laboratory submission
│
├── src/                                                      # Modular, production-grade Python engine
│   ├── checkpointing.py                                      # Atomic JSON/JSONL state persistence & resume engine
│   ├── cv.py                                                 # Universal rolling-origin walk-forward CV generator
│   ├── device.py                                             # Hardware detection (PyTorch CUDA, cuDF, cuML)
│   ├── evaluation.py                                         # Vectorized metrics engine (MAE, RMSE, Coverage)
│   └── pipelines.py                                          # BaseForecastPipeline, Naive, AR, ARIMA, SARIMA
│
├── data/                                                     # Data management & cache layer
│   ├── raw/
│   │   ├── crime_extract.csv                                 # Raw Chicago Data Portal extract (2.39 GB, 8.62M rows)
│   │   └── source_pointer.json                               # Data provenance metadata & source URLs
│   └── processed/
│       ├── cleaned_raw.parquet                               # Compressed columnar storage (86.16 MB, 8.62M rows)
│       ├── cleaned_raw_metadata.json                         # Chunk processing audit & ingestion log
│       ├── series_location1.csv                              # District 008 full weekly series (1,339 periods)
│       ├── train_location1.csv / test_location1.csv          # District 008 chronological split (1,327 / 12)
│       ├── series_location2.csv                              # District 011 full weekly series (1,339 periods)
│       ├── train_location2.csv / test_location2.csv          # District 011 chronological split (1,327 / 12)
│       └── series_location1_theft.csv                        # Disaggregated THEFT category series (1,339 periods)
│
├── models/                                                   # Serialized production pipelines
│   └── D1_chicago/
│       └── location2/
│
├── artifacts/                                                # Reproducibility audit trail, manifests & models
│   └── D1_chicago/
│       ├── ar_pipeline.joblib                                # Fitted AR(3) pipeline artifact
│       ├── arima_pipeline.joblib                             # Fitted ARIMA(2,1,0) pipeline artifact
│       ├── naive_pipeline.joblib                             # Fitted Naive persistence pipeline
│       ├── final_ar3_pipeline.joblib                         # Final audited AR(3) model bundle
│       ├── final_arima2_1_0_pipeline.joblib                  # Final audited ARIMA(2,1,0) model bundle
│       ├── final_naive_pipeline.joblib                       # Final audited Naive model bundle
│       ├── lstm_model.pt                                     # PyTorch state_dict for deep LSTM forecaster
│       ├── governance_manifest.json                          # Ethical boundary, PII audit & license terms
│       ├── checkpoints/                                      # Resumable execution state logs (JSONL)
│       │   ├── arima_order_search.jsonl                      # Grid search history across 32 orders
│       │   ├── ar_rolling_origin.jsonl                       # 83-fold AR backtest records
│       │   ├── arima_rolling_origin.jsonl                    # 83-fold ARIMA backtest records
│       │   └── sarima_rolling_origin.jsonl                   # 83-fold SARIMA backtest records
│       ├── location2/                                        # District 011 replication artifacts
│       │   └── checkpoints/                                  # District 011 order search & CV checkpoints
│       └── D1_chicago/
│           ├── manifest.json                                 # Master experiment manifest & parameters
│           └── file_inventory.json                           # 74-file cryptographic inventory & hashes
│
└── results/                                                  # Metric tables, CSV outputs & high-res figures
    └── D1_chicago/
        ├── baseline_scores.json                              # Naive baseline holdout metrics
        ├── ar_results.json                                   # AR(3) coefficients & holdout scores
        ├── arima_summary.json                                # ARIMA(2,1,0) candidate summary & improvements
        ├── arima_candidate_ranking.csv                       # Full 32-configuration grid search results
        ├── arima_test_predictions.csv                        # Point forecasts & intervals vs actuals
        ├── test_predictions.csv                              # Comparative predictions across all models
        ├── model_comparison.csv                              # Holdout metrics summary table
        ├── final_model_comparison.csv                        # Consolidated holdout & 83-fold CV metrics
        ├── rolling_origin_results.csv                        # 83-fold walk-forward performance log
        ├── residuals_ljung_box.csv                           # Portmanteau white-noise test statistics
        │
        ├── figures/ (Primary District 008):
        │   ├── fig_raw_location_counts.png                   # Top 20 police district distribution
        │   ├── fig_raw_volume_over_time.png                  # 25-year monthly citywide temporal trend
        │   ├── fig_series_location1.png                      # District 008 weekly series (2001–2026)
        │   ├── fig_train_test_split.png                      # In-sample vs locked holdout visualization
        │   ├── fig_rolling_stats.png                         # 52-week rolling mean and rolling std
        │   ├── fig_acf_pacf.png                              # Level series autocorrelation correlogram
        │   ├── fig_acf_pacf_diff.png                         # Differenced series autocorrelation correlogram
        │   ├── fig_actual_vs_naive.png                       # Naive persistence holdout forecast
        │   ├── fig_actual_vs_ar.png                          # AR(3) holdout forecast vs actuals
        │   ├── fig_actual_vs_arima.png                       # ARIMA(2,1,0) forecast with 95% intervals
        │   ├── fig_final_forecast_comparison.png             # Multi-model holdout comparative plot
        │   ├── fig_rolling_origin_mae.png                    # 83-fold backtesting error distribution
        │   ├── fig_rolling_origin_all_models.png             # AR vs ARIMA vs SARIMA fold-by-fold errors
        │   ├── fig_residuals_over_time.png                   # In-sample standardized residual trace
        │   ├── fig_residuals_acf.png                         # Autocorrelation of model residuals
        │   ├── fig_structural_break_halves.png               # Early (2001–2013) vs Late (2014–2026) shift
        │   └── fig_final_model_comparison.png                # Final comparative error bar chart
        │
        ├── location2/ (Replication District 011):
        │   ├── baseline_scores.json, ar_results.json, arima_summary.json
        │   ├── rolling_origin_combined.csv, residuals_ljung_box.csv
        │   └── fig_*.png (9 complete replication plots)
        │
        └── D1_chicago/
            ├── location_comparison.csv                       # Statistical contrast between Districts 008 and 011
            └── advanced/
                ├── multi_location_results.csv                # 21-district batch ARIMA evaluation
                ├── multi_location_cpu.jsonl                  # Per-district checkpoint log
                └── fig_multi_location_distributions.png      # Cross-district error & volume distributions
```

---

## 3. Dataset Ingestion, Provenance & Governance

### 3.1 Data Provenance & Governance Card
* **Primary Source Agency**: City of Chicago Data Portal (Department of Police).
* **Dataset Identifier**: `Crimes - 2001 to Present` (`ijzp-q8t2`).
* **Licensing**: City of Chicago Open Data License (Public Domain / Attribution).
* **Temporal Extent**: January 1, 2001 through August 24, 2026 (25 years, 8 months).
* **Record Count**: 8,623,069 incident occurrences.
* **Raw Size on Disk**: 2.39 GB CSV.
* **Update Frequency**: Daily (subject to 7-day administrative lag).
* **Governance Audit**: Full verification conducted (`artifacts/D1_chicago/governance_manifest.json`). Zero direct PII is stored; case numbers serve solely for deduplication.

### 3.2 Ingestion Engine & Columnar Optimization
Reading 8.62M records repeatedly during iterative experimentation is computationally prohibitive. Stage 1 implements a resumable chunked ingestion engine:
* **Chunk Processing**: Streamed the 2.39 GB raw CSV in 18 iterative chunks using memory-mapped buffers.
* **Column Projection**: Filtered out auxiliary geographic text fields, retaining only the 4 canonical attributes: `Date`, `District`, `Case Number`, `Primary Type`.
* **Deduplication Policy**: Dropped duplicate records by `Case Number`. Bad or unparseable timestamps: **0 dropped** (100% timestamp fidelity).
* **Optimized Storage**: Serialized to Apache Parquet (`cleaned_raw.parquet`, snappy compression). Storage footprint was reduced from **2,397 MB to 86.16 MB (96.4% reduction)** while slashing loading times from ~45 seconds to under 0.8 seconds.

```text
Raw CSV (2.39 GB, 8.62M Rows)
      │
      ├──> Chunked Streaming Ingest (18 Chunks, usecols=[Date, District, Case Number, Primary Type])
      ├──> Timestamp Normalization (pd.to_datetime with UTC alignment)
      ├──> Primary Key Deduplication (subset=['Case Number'])
      │
      └──> Snappy Compressed Parquet (86.16 MB, 8,623,069 Rows) [0.8s load time]
```

### 3.3 Spatial Feature Separation Principle
A critical architectural decision is addressed in Stage 1.2: **Why location defines an isolated time series rather than a numeric feature in a unified model.**
1. **False Metric Topology**: District IDs (`001` through `025`) are discrete administrative administrative partitions. Feeding district as a numeric regressor falsely imposes metric ordering (e.g., implying District 8 is twice District 4, or that District 7 is adjacent to District 8).
2. **Spatial Heterogeneity**: Criminal incident dynamics are driven by highly localized socioeconomic infrastructure, beat structures, and reporting tendencies. The autoregressive coefficients of District 008 ($\phi_1 = 0.471$) differ fundamentally from District 011 ($\phi_1 = 0.385$).
3. **Leakage & Independence**: Maintaining distinct univariate pipelines prevents global cross-series variance leakage and ensures modular retraining per police beat.

---

## 4. Preprocessing, Feature Construction & Leakage Control

### 4.1 Temporal Resampling & Missing Value Semantics
Raw records represent individual, point-in-time incident dispatches. Modeling aggregate weekly volume requires regular temporal resampling:
* **Resampling Cadence**: `W-MON` (Weekly, anchored every Monday).
* **Index Monotonicity**: Verified strict ascending temporal monotonicity ($t_0 < t_1 < \dots < t_N$) with zero duplicate timestamps.
* **Total Continuous Periods**: 1,339 weekly periods spanning `2001-01-01` to `2026-08-24`.
* **Zero-Filling Semantic Justification**: The series is constructed via `.resample('W-MON').size().asfreq('W-MON', fill_value=0)`. In incident analytics, a period with zero records represents **zero reported incidents**, not missing data. Imputing via interpolation or mean-filling would fabricate non-existent crime. Setting `fill_value=0` preserves factual administrative ground truth.

### 4.2 Leakage-Safe Chronological Partitioning
In time-series forecasting, traditional random $k$-fold cross-validation is strictly invalid because future observations would leak into historical parameter estimation. 

* **Holdout Design**: A strict chronological split isolates the final **12 weeks** (~3 calendar months: `2026-06-08` to `2026-08-24`) as a locked operational holdout.
* **In-Sample Partition**: The initial **1,327 weeks** (`2001-01-01` to `2026-06-01`) are reserved for diagnostic testing, order selection, and parameter fitting.
* **Isolation Wall**: All statistical transformations, ADF unit-root tests, differencing orders, AIC grid searches, and parameter estimations are calculated strictly within the 1,327-week in-sample partition. The test partition is touched exactly once during final locked evaluation.

```
0                                                                 1327            1339 Weeks
+-------------------------------------------------------------------+---------------+
|                     IN-SAMPLE TRAINING PARTITION                  |  LOCKED TEST  |
|                       1,327 Weeks (2001-01-01 -> 2026-06-01)      |   12 Weeks    |
|   Used for: Stationarity Checks, ACF/PACF, Grid Search, CV, Fits  | Holdout Only  |
+-------------------------------------------------------------------+---------------+
                                                                    ^ Chronological Split
```

---

## 5. Exploratory Data Analysis & Time-Series Diagnostics

All diagnostic procedures were executed on the in-sample training partition (`train_location1.csv`).

### 5.1 Geographic & Municipal Volume Distribution
Analyzing citywide district incident counts reveals high geographic concentration across Chicago's 25 police districts:
* **Top District**: **District 008 (Chicago Lawn)** registered the highest historical incident volume with **577,431 records** (mean weekly volume: 431.24 incidents).
* **Second District**: **District 011 (Harrison)** followed closely with **545,232 records** (mean weekly volume: 407.19 incidents).
* **Selection Rationale**: Districts 008 and 011 were selected as Primary and Replication locations due to their high statistical support and distinct urban profiles.

### 5.2 Stationarity Diagnostics & Unit Root Analysis
Prior to fitting Autoregressive or Moving Average processes, stationarity was formally evaluated using the Augmented Dickey-Fuller (ADF) test.

```text
Augmented Dickey-Fuller Test on District 008 (In-Sample Level Series):
  ADF Test Statistic : -1.4148
  p-value            : 0.5751 (Fail to Reject H0: Unit Root Present)
  Critical Values    : 1%: -3.435, 5%: -2.864, 10%: -2.568
  Conclusion         : Non-Stationary. Exhibits stochastic downward drift. Differencing d=1 required.

Augmented Dickey-Fuller Test on First-Differenced Series (d=1):
  ADF Test Statistic : -14.9213
  p-value            : 1.42e-27 (Reject H0: Series is Stationary)
  Conclusion         : Differencing achieves weak stationarity.
```

### 5.3 Rolling Statistical Dynamics & Correlograms
* **Rolling Statistics (52-Week Window)**: 
  * The 52-week moving mean drifts monotonically downward from ~600 incidents/week in 2001 to ~300 incidents/week in 2026.
  * Rolling standard deviation fluctuates between 40 and 80 incidents/week with regular annual periodicity, confirming seasonal heteroscedasticity.
* **Autocorrelation Function (ACF)**: 
  * Level series ACF shows extremely slow geometric decay across 50+ lags, characteristic of a non-stationary integrated process ($d \ge 1$) coupled with annual seasonality at lag 52.
* **Partial Autocorrelation Function (PACF)**:
  * First-differenced PACF cuts off sharply after lag 2 and lag 3, providing theoretical justification for candidate autoregressive orders $p \in \{2, 3\}$.

---

## 6. Model Architecture & Pipeline Specifications

All estimators are encapsulated within an object-oriented scikit-learn compatible architecture inheriting from `BaseForecastPipeline` (`src/pipelines.py`).

### 6.1 Baseline: Naive (Persistence) Pipeline
The simplest benchmark assumes future values equal the most recent observed incident count:
$$\hat{Y}_{T+h} = Y_T \quad \forall h \in \{1, \dots, H\}$$
* Stores $Y_{1327} = 308.0$ (week ending 2026-06-01).
* Serves as the minimum performance threshold; any model failing to beat Naive is rejected as operationally unviable.

### 6.2 Autoregressive Pipeline: AR(3)
Formulated with lag order $p=3$ and a deterministic linear time trend (`trend='ct'`):
$$\hat{Y}_t = c + \beta t + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \phi_3 Y_{t-3} + \epsilon_t$$

* **Estimated In-Sample Parameters**:
  * Constant ($c$): **68.1760**
  * Time Trend ($\beta$): **-0.0350** (confirms long-term secular decline)
  * Lag 1 ($\phi_1$): **0.4706** ($t = 16.88, p < 10^{-15}$)
  * Lag 2 ($\phi_2$): **0.2146** ($t = 6.94, p < 10^{-11}$)
  * Lag 3 ($\phi_3$): **0.2107** ($t = 7.55, p < 10^{-13}$)
* **Root Stability**: $\sum \phi_i = 0.4706 + 0.2146 + 0.2107 = 0.8959 < 1.0$, satisfying characteristic polynomial stability conditions.

### 6.3 ARIMA Pipeline with Checkpointed Grid Search
An integrated autoregressive moving average model with differencing order $d=1$:
$$\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d Y_t = c + \left(1 + \sum_{j=1}^q \theta_j B^j\right) \epsilon_t$$

* **Checkpointed Hyperparameter Search**: Evaluated 32 distinct $(p, d, q) \times \text{trend}$ combinations via atomic JSONL logging (`artifacts/D1_chicago/checkpoints/arima_order_search.jsonl`).
* **Selection Criterion**: Validation MAE on rolling holdouts was primary; Akaike Information Criterion (AIC) served as tie-breaker.

```text
Top 5 ARIMA Configurations from In-Sample Grid Search:
Rank  Order      Trend  Validation MAE  Validation RMSE  AIC (Train Fit)
 1.   [2, 1, 0]    n        20.90            27.41          13,271.25  <-- Selected Champion
 2.   [3, 1, 0]    n        20.92            27.62          13,237.40
 3.   [2, 1, 0]    t        20.94            27.46          13,273.25
 4.   [3, 1, 0]    t        21.08            27.80          13,239.39
 5.   [4, 1, 0]    n        21.12            27.87          13,234.64
```
* **Champion Model**: **ARIMA(2, 1, 0)** with no constant trend (`trend='n'`). Differencing eliminated the need for an explicit drift parameter.

### 6.4 Seasonal ARIMA: SARIMA(2, 1, 0) $\times$ (1, 1, 1)$_{52}$
To account for annual summer incident surges, seasonal parameters with period $s=52$ were incorporated:
$$(1 - \phi_1 B - \phi_2 B^2)(1 - \Phi_1 B^{52}) (1 - B)(1 - B^{52}) Y_t = (1 + \Theta_1 B^{52}) \epsilon_t$$
* Evaluated after confirming training length ($1,327 \text{ weeks} \gg 2 \times 52 = 104 \text{ weeks}$).

### 6.5 Deep Recurrent Network: PyTorch LSTM Forecaster
* **Architecture**: 2-layer stacked Long Short-Term Memory network, hidden dimension = 64, dropout = 0.10.
* **Input Representation**: Sliding lookback window of 12 weeks ($t-11 \dots t$) mapped to $t+1$.
* **Optimization**: Trained for 200 epochs using Adam ($\eta = 10^{-3}$, MSE loss, gradient clipping $\Vert g \Vert_2 \le 1.0$) with checkpoint resumption (`artifacts/D1_chicago/lstm_model.pt`).

---

## 7. Experimental Results & Backtesting Analysis

### 7.1 Locked 12-Week Holdout Evaluation (District 008)

The models were evaluated against actual incident occurrences over the 12 locked test weeks (`2026-06-08` to `2026-08-24`).

```text
Comparative Trajectory on Locked 12-Week Test Horizon:
Date         Actual    Naive Pred    AR(3) Pred    ARIMA(2,1,0)    ARIMA 95% Confidence Interval
2026-06-08     279        308.0         292.0          302.3             [228.6 , 376.0]
2026-06-15     256        308.0         283.5          296.9             [214.3 , 379.5]
2026-06-22     295        308.0         282.6          301.1             [211.0 , 391.1]
2026-06-29     272        308.0         276.9          300.4             [199.3 , 401.6]
2026-07-06     296        308.0         272.2          299.6             [190.3 , 409.0]
2026-07-13     303        308.0         268.5          300.2             [183.4 , 417.0]
2026-07-20     274        308.0         264.6          300.1             [175.9 , 424.4]
2026-07-27     263        308.0         260.9          300.0             [168.9 , 431.2]
2026-08-03     254        308.0         257.5          300.1             [162.5 , 437.7]
2026-08-10     238        308.0         254.3          300.1             [156.2 , 444.0]
2026-08-17     237        308.0         251.2          300.1             [150.2 , 449.9]
2026-08-24      86*       308.0         248.3          300.1             [144.5 , 455.7]
*Note: Final week reflects Chicago Data Portal's 7-day administrative publishing cutoff.
```

| Model Rank & Architecture | Locked Test MAE | Locked Test RMSE | Rel. Gain vs Naive (MAE) | Rel. Gain vs Naive (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| 1. **AR(3)** | **26.99** | **49.81** | **+49.63%** | **+35.01%** |
| 2. **SARIMA(2, 1, 0) $\times$ (1, 1, 1)$_{52}$** | 34.51 | 63.73 | +35.60% | +16.85% |
| 3. **PyTorch LSTM** | 36.08 | 60.21 | +32.66% | +21.45% |
| 4. **ARIMA(2, 1, 0)** | 46.13 | 71.25 | +13.91% | +7.04% |
| 5. **Naive (Persistence)** | 53.58 | 76.65 | Baseline | Baseline |

### 7.2 83-Fold Rolling-Origin Walk-Forward Backtesting (Universal CV)
Relying on a single 12-week test window introduces severe sample selection bias. To rigorously test out-of-sample stability, an 83-fold expanding walk-forward cross-validation protocol was executed across 13 years of operational data:
* **Initial Window ($T_{init}$)**: 668 weeks (spanning `2001-01-01` to `2013-10-21`).
* **Forecast Horizon ($H$)**: 8 weeks.
* **Step Size**: 8 weeks (advancing the forecast origin 83 consecutive times).

```
Fold 00: [======== Train: 668 Wks ========] -> [Test: 8 Wks]
Fold 01: [========== Train: 676 Wks ==========] -> [Test: 8 Wks]
...
Fold 82: [==================== Train: 1,324 Wks ====================] -> [Test: 8 Wks]
```

```text
83-Fold Rolling-Origin Cross-Validation Results:
Model Architecture              Mean MAE    Std MAE    Mean RMSE    Std RMSE    Folds Evaluated
ARIMA(2, 1, 0)                   27.87       12.74       32.38        14.09           83
SARIMA(2, 1, 0)x(1, 1, 1, 52)    29.16       11.87       34.90        13.86           83
AR(3)                            30.14       13.78       35.32        15.05           83
```

#### Methodological Discovery: Resolving the "Holdout Paradox"
* **The Conflict**: On the single 12-week holdout, AR(3) scored MAE 26.99 while ARIMA(2, 1, 0) scored 46.13.
* **The Root Cause**: The single test holdout coincided with a sharp late-summer downward dip. AR(3)'s deterministic trend parameter ($\beta = -0.035$) caused it to decay downwards rapidly, accidentally matching this specific 12-week dip.
* **The Generalization Proof**: Across 83 independent historical test folds, **ARIMA(2, 1, 0)** achieved lower mean error (MAE **27.87** vs **30.14**) and lower error variance ($\sigma = 12.74$ vs $13.78$). ARIMA generalises better across varying regimes, proving the necessity of multi-fold walk-forward validation over single-slice testing.

### 7.3 Prediction Interval Calibration
ARIMA(2, 1, 0) generated 95% nominal parametric prediction intervals over the holdout horizon:
* **Empirical Coverage**: **91.7%** (11 of 12 test points fell strictly within the upper and lower bounds).
* **Uncertainty Expansion**: Interval width expanded monotonically from $\pm 73.7$ incidents at $h=1$ to $\pm 155.6$ incidents at $h=12$, accurately reflecting compound forecast variance. The sole breach occurred at $h=12$ due to the administrative reporting drop.

---

## 8. Residual Diagnostics & Model Adequacy

To ensure all systematic temporal structure was captured, the in-sample residuals of the champion ARIMA(2, 1, 0) model were audited.

```text
Ljung-Box Portmanteau Test on ARIMA(2, 1, 0) In-Sample Residuals:
Lag Order (m)    Q-Statistic    p-value      Null Hypothesis (H0: White Noise)
   Lag 1            1.15         0.283       Fail to Reject (Clean White Noise)
   Lag 2            2.14         0.342       Fail to Reject (Clean White Noise)
   Lag 5           26.04         8.76e-05    Reject H0 (Serial Correlation Present)
   Lag 10          34.22         1.70e-04    Reject H0
   Lag 20          53.80         6.20e-05    Reject H0
   Lag 52         110.91         7.07e-05    Reject H0 (Annual Seasonality Detected)
```

### 8.1 Diagnostic Interpretation
1. **Short-Term Independence**: Residuals at lags 1 and 2 pass the white-noise test ($p > 0.28$), confirming that the AR order $p=2$ and differencing $d=1$ fully removed short-range autocorrelation.
2. **Seasonal Residual Structure**: Rejection of the white-noise null at higher lags ($m \ge 5$ and $m = 52$) indicates residual cyclicality. This diagnostic directly motivated the SARIMA extension in Stage 14a, which models the 52-week lag and eliminates this residual distortion.

---

## 9. Spatial Replication: District 011 (Harrison)

To assess spatial generalizability, the entire diagnostic and modeling pipeline was replicated under an identical protocol on Chicago's second highest-volume jurisdiction: **District 011 (Harrison)**.

```text
Spatial Contrast: District 008 (Chicago Lawn) vs. District 011 (Harrison):
Metric / Attribute               District 008 (Primary)       District 011 (Replication)
Total Incident Records           577,431                      545,232
Continuous Weekly Periods        1,339 Weeks                  1,339 Weeks
Mean Weekly Incident Count       431.24 incidents             407.19 incidents
Weekly Standard Deviation        143.38 incidents             119.76 incidents
In-Sample ADF Test (Level)       p = 0.575 (Non-Stationary)   p = 0.108 (Near-Stationary)
Selected Differencing Order (d)  d = 1                        d = 0
Selected ARIMA Specification     ARIMA(2, 1, 0), trend='n'    ARIMA(2, 0, 0), trend='c'
Locked Holdout Naive MAE         53.58                        43.17
Locked Holdout AR(3) MAE         26.99 (Best)                 35.37 (Best)
Locked Holdout ARIMA MAE         46.13                        59.17
83-Fold Walk-Forward Best        ARIMA(2, 1, 0) (MAE 27.87)   AR(3) (MAE 28.84)
```

### 9.1 Data-Grounded Spatial Insights
1. **Divergent Integration Orders**: While District 008 exhibited strong stochastic non-stationarity requiring first-differencing ($d=1$), District 011 showed weaker unit-root evidence ($p = 0.108$). Grid search selected $d=0$ with a constant drift term (`ARIMA(2, 0, 0)`).
2. **Historical Volatility Anomaly**: District 011 suffered from an extreme localized crime surge in the early 2000s (visible in `fig_rolling_stats.png`), which inflated overall series variance and degraded ARIMA's single-holdout tracking.
3. **End-of-Extract Coverage Artefact**: Both locations exhibited an abrupt collapse in incident volume during the final recorded week (`2026-08-24`: District 008 fell to 86; District 011 fell to 65). This is an administrative reporting artifact from the City of Chicago's 7-day data pipeline, not an operational drop in criminal activity.

---

## 10. Advanced Extensions & Comparative Benchmarks

### 10.1 Stage 14a — Seasonal ARIMA (SARIMA)
Incorporating annual seasonality $(2, 1, 0) \times (1, 1, 1)_{52}$ produced a major breakthrough on the locked holdout:
* **Holdout Score**: Slashed test MAE from **46.13 down to 34.51** (a 25.2% error reduction over standard ARIMA).
* **83-Fold Stability**: Maintained consistent performance across all backtest folds (Mean MAE: **29.16 ± 11.87**), validating that crime in Chicago is fundamentally governed by an annual 52-week seasonal wave.

### 10.2 Stage 14b — Category-Specific Disaggregation (THEFT)
Isolated reported **THEFT** occurrences in District 008 (`series_location1_theft.csv`):
* **Summary Stats**: Mean weekly volume = **83.92 ± 30.10** incidents (min: 6, max: 169).
* **ADF Test**: Test stat = -2.55, $p = 0.104$.
* **Holdout Results**: AR(3) scored **MAE 9.33 / RMSE 13.21** vs. Naive MAE 17.92 (a **47.9% improvement**). Disaggregated property crime displays smoother autocorrelation dynamics than aggregate violent crime.

### 10.3 Stage 14c — Structural-Break Analysis
Split District 008 into two equal temporal halves to evaluate non-stationarity of the data-generating process:
* **Early Half (`2001-01-01` to `2013-10-21`)**: Mean = **500.25**, Std = **83.43**.
* **Late Half (`2013-10-28` to `2026-08-24`)**: Mean = **314.27**, Std = **66.56**.
* **Structural Shift**: Secular drop of **-185.98 incidents/week** with a variance ratio of **0.64**. This permanent structural shift confirms why static regression models fail and why rolling-origin retraining is mandatory for municipal operations.

### 10.4 Stage 14d — Multi-District Scaling (21 Police Districts)
Scaled the ARIMA pipeline across all 21 Chicago police districts with volume exceeding 200,000 records using a unified rectangular data matrix $(1,339 \times 21)$:
* **Aggregate Performance**: Citywide average test MAE = **35.05 ± 13.13**, average RMSE = **55.39 ± 16.48**.
* **Top Performing District**: District 16 (Jefferson Park) achieved the lowest forecast error: **MAE 18.66, RMSE 36.26** (mean volume: 217.2/week).
* **Highest Error District**: District 12 (Near West) recorded **MAE 81.81, RMSE 107.27** due to high commercial density and variable event spikes.
* **Hardware Acceleration Benchmark**: Evaluated RAPIDS cuML batched GPU ARIMA vs CPU statsmodels loop. CPU fallback processed 21 districts in 15.6 seconds (~0.72s/district).

### 10.5 Stage 14e — Deep Learning Verdict (PyTorch LSTM vs. ARIMA)
* **Holdout Scores**: PyTorch LSTM achieved test MAE **36.08** and RMSE **60.21** (beating ARIMA(2,1,0)'s holdout MAE of 46.13).
* **Critical Statistical Verdict**: While the LSTM beat ARIMA on the single 12-week slice by ~10 MAE points, this gap is **smaller than ARIMA's own backtesting standard deviation ($\sigma = 12.74$)**. The apparent improvement falls within normal temporal noise.
* **Operational Drawbacks of Deep Learning**:
  1. **Compute Overhead**: Required 200 epochs of iterative neural optimization vs. closed-form analytical estimation for AR/ARIMA.
  2. **Uncertainty Quantification**: Standard LSTMs lack parametric prediction intervals, whereas ARIMA provides mathematically rigorous 95% confidence intervals essential for municipal budgeting.

---

## 11. Governance, Ethical Boundaries & Responsible-Use

Predictive modeling of criminal incident records entails significant societal risks. This project enforces strict governance guidelines in accordance with the course specification:

```
+----------------------------------------------------------------------------------------------------+
|                                RESPONSIBLE-USE BOUNDARY PROTOCOL                                   |
+----------------------------------------------------------------------------------------------------+
| 1. REPORTED INCIDENTS != TRUE CRIME PREVALENCE                                                     |
|    Reported records measure administrative police dispatch activity, not true underlying           |
|    criminal activity. Reporting rates reflect community trust and police deployment patterns.      |
+----------------------------------------------------------------------------------------------------+
| 2. STRICTLY AGGREGATE SPATIAL UNITS (DISTRICT LEVEL ONLY)                                          |
|    Predictions are restricted to high-level police district weekly totals. Address-level,          |
|    block-level, or individual-level forecasting is explicitly prohibited by design.                |
+----------------------------------------------------------------------------------------------------+
| 3. PROHIBITION OF PERSON-LEVEL PROFILING & AUTOMATED PATROL DISPATCH                              |
|    Models must never generate individual risk scores or be used for autonomous punitive actions.   |
|    The system is intended solely for municipal workload planning and non-punitive support.         |
+----------------------------------------------------------------------------------------------------+
| 4. PREVENTING ALGORITHMIC RUNAWAY FEEDBACK LOOPS                                                   |
|    Deploying police based on raw incident forecasts inflates reporting rates in those zones,       |
|    creating self-fulfilling prediction loops. Forecasts must be interpreted alongside context.     |
+----------------------------------------------------------------------------------------------------+
```

---

## 12. Hardware, Reproducibility & Execution Guide

### 12.1 Hardware Acceleration & Environment
* **Python Runtime**: Python 3.13.15 (64-bit AMD64) on Windows 11.
* **Core Libraries**: `statsmodels >= 0.14.0`, `scikit-learn >= 1.4.0`, `torch >= 2.2.0`, `pandas >= 2.2.0`, `pyarrow >= 15.0.0`, `joblib >= 1.3.0`.
* **Hardware Detection**: `device.py` detects CUDA GPUs and RAPIDS libraries (`cudf`, `cuml`). The CPU fallback branch executes seamlessly across all architectures.

### 12.2 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Raunak-23/Predictive_Analytics.git
cd Predictive_Analytics/Crime_Indicent_Analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 12.3 Running the Forecasting Engine
1. **Interactive Execution**: Launch JupyterLab to inspect the complete 159-cell narrative:
   ```bash
   jupyter lab notebooks/chicago_crimes.ipynb
   ```
2. **Headless Execution & HTML Export**:
   ```bash
   jupyter nbconvert --to html --execute notebooks/chicago_crimes.ipynb
   ```
3. **Resuming Checkpointed Runs**: If grid search or walk-forward backtesting is interrupted, the engine resumes automatically from the last atomic record in `artifacts/D1_chicago/checkpoints/` without recomputing earlier folds.

### 12.4 Reproducibility Audit & Manifest Verification
All 74 experiment artifacts, fitted pipelines, and metric tables are cataloged in `artifacts/D1_chicago/D1_chicago/manifest.json`. The reproducibility suite can be verified via:
```bash
python -c "
import json
with open('artifacts/D1_chicago/D1_chicago/manifest.json') as f:
    m = json.load(f)
print('Verified experiment stage:', m['stage'])
print('District 008 Test MAE:', m['location_1']['locked_test_mae'])
print('District 011 Test MAE:', m['location_2']['locked_test_mae'])
"
```

---

## 13. Academic Integrity & Acknowledgments

This research and software implementation was produced for **MDI3003 Advanced Predictive Analytics** at **Vellore Institute of Technology (VIT), Vellore**.

* **Author**: Raunak Pal
* **Data Provider**: City of Chicago Data Portal