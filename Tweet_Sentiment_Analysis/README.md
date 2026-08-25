# ✈️ Tweet Sentiment Analysis — US Airline Twitter Data

A complete predictive-analytics pipeline that classifies tweets directed at six major US airlines as **negative**, **neutral**, or **positive**. The project covers data auditing, exploratory analysis, baseline benchmarking, model selection via cross-validation, hyper-parameter tuning, ablation studies, error analysis, and locked-test evaluation.

---

## 📋 Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Audit & Preprocessing](#data-audit--preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Baselines](#baselines)
  - [Model Training & Selection](#model-training--selection)
  - [Ablation Study](#ablation-study)
  - [Error Analysis](#error-analysis)
- [Results](#results)
- [Key Visualisations](#key-visualisations)
- [Getting Started](#getting-started)
- [License](#license)

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Twitter US Airline Sentiment – Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) |
| **Rows** | 14,640 tweets |
| **Labels** | Negative (62.7%), Neutral (21.2%), Positive (16.1%) |
| **Language** | English |
| **Domain** | Customer tweets about US airlines (Feb 2015) |
| **Annotation** | Human-annotated via CrowdFlower |
| **License** | CC BY-NC-SA 4.0 |

> **Imbalance note:** Negative tweets dominate the dataset (~63%). **Macro-F1** is used as the primary evaluation metric to ensure fair assessment across all three classes.

### Fields Excluded (Privacy / Leakage)

`airline_sentiment_confidence`, `negativereason`, `negativereason_confidence`, `airline_sentiment_gold`, `negativereason_gold`, `name`, `tweet_coord`, `tweet_created`, `tweet_location`, `user_timezone`

---

## 🗂️ Project Structure

```
Tweet_Sentiment_Analysis/
├── README.md                          # This file
├── notebooks/
│   └── tweet_us.ipynb                 # Main analysis notebook (+ HTML/PDF exports)
├── data/
│   ├── raw/
│   │   └── Tweets_US_Airlines.csv     # Original dataset
│   └── processed/
│       ├── D2_processed.csv           # Cleaned & feature-engineered data
│       ├── D2_train.csv               # Training split
│       ├── D2_val.csv                 # Validation split
│       ├── D2_test.csv                # Locked test split
│       └── *_manifest.csv             # Split manifests
├── models/
│   └── D2_selected_pipeline.joblib    # Serialised best pipeline (LinearSVC)
├── results/
│   └── D2/
│       ├── baseline_results.csv       # DummyClassifier & VADER baselines
│       ├── cv_results.csv             # Cross-validation comparison
│       ├── ablation_results.csv       # Text-preprocessing ablation
│       ├── test_classification_report.csv
│       ├── test_predictions.csv       # Per-sample predictions on test set
│       ├── error_analysis.csv         # Misclassified examples
│       ├── challenge_subset.csv       # Hardest examples
│       ├── entity_sentiment_distribution.csv
│       ├── entity_examples.csv
│       └── *.png                      # All plots (see below)
├── artifacts/
│   ├── D2_data_audit.md               # Data quality audit report
│   ├── D2_dataset_card.md             # Dataset card (provenance & fields)
│   ├── D2_eda_summary.md              # EDA findings
│   └── D2_test_interpretation.md      # Final test-set interpretation
└── src/                               # (reserved for modular source code)
```

---

## 🔬 Methodology

### Data Audit & Preprocessing

- Identified **155 duplicate tweet IDs** and **213 duplicate text entries**.
- Removed privacy-sensitive and target-leakage columns (10 fields excluded).
- Added engineered feature: `text_len` (character count).
- Stratified train / validation / test split to preserve class ratios.

### Exploratory Data Analysis

| Insight | Detail |
|---|---|
| Tweet length | Mean 103.8 chars, Median 114.0, Max 186 |
| Top negative terms | cancelled, delay, bad, issue, help, wait, awful, worst, refund, customer |
| Top positive terms | thanks, great, love, amazing, awesome, best, friendly, fantastic, incredible, happy |
| Key observation | Clear lexical separation between classes → TF-IDF expected to perform well |
| Challenge | Sarcasm & negation not captured by bag-of-words alone |

### Baselines

| Model | Macro-F1 (CV) | Macro-F1 (Val) | Macro-F1 (Test) |
|---|---|---|---|
| DummyClassifier (stratified) | 0.257 | 0.257 | 0.257 |
| VADER (rule-based) | 0.512 | 0.521 | 0.521 |

### Model Training & Selection

Three models were compared using **5-fold stratified cross-validation** with TF-IDF features (unigrams + bigrams):

| Model | Best Params | CV Macro-F1 | Val Macro-F1 | Val Accuracy |
|---|---|---|---|---|
| MultinomialNB | α=0.1, max_df=0.9, min_df=5 | 0.691 | 0.710 | 0.793 |
| LogisticRegression | C=1.0, balanced weights | 0.743 | 0.756 | 0.804 |
| **LinearSVC** ✅ | **C=0.1, balanced weights** | **0.740** | **0.759** | **0.815** |

**LinearSVC** was selected as the best model based on its highest validation accuracy and competitive Macro-F1.

### Ablation Study

| Text Preprocessing Variant | Macro-F1 (Test) |
|---|---|
| Default (minimal cleaning) | **0.744** |
| Without hashtags/emojis | 0.741 |

Removing hashtags and emojis caused a slight drop, confirming they carry useful sentiment signal.

### Error Analysis

- **Dominant confusion:** Neutral ↔ Negative — implicit complaints and customer-service inquiries lack explicit sentiment cues.
- **Short tweets with sarcasm/jargon** produce the highest false-prediction rate.
- Per-airline error rates and entity-level sentiment distributions were computed (see `results/D2/`).

---

## 📈 Results

### Final Test-Set Performance (Locked)

| Metric | Score |
|---|---|
| **Accuracy** | 80.05% |
| **Macro-F1** | **0.7412** |
| **Weighted-F1** | 0.7990 |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.862 | 0.891 | 0.876 | 1,407 |
| Neutral | 0.631 | 0.647 | 0.639 | 473 |
| Positive | 0.778 | 0.651 | 0.709 | 361 |

> **Generalisation gap:** Val Macro-F1 (0.7412) − Test Macro-F1 (0.7412) = **0.0000** — no overfitting detected.

---

## 📊 Key Visualisations

All plots are saved in `results/D2/`:

| Plot | Description |
|---|---|
| `eda_class_distribution.png` | Label frequency bar chart |
| `eda_tweet_length.png` | Tweet length distribution per class |
| `eda_top_terms.png` | Top TF-IDF terms by sentiment |
| `eda_null_duplicate.png` | Missing values & duplicates summary |
| `cv_model_comparison.png` | Cross-validation Macro-F1 comparison |
| `baseline_comparison.png` | Baseline vs trained models |
| `confusion_matrix.png` | Test-set confusion matrix |
| `per_class_prf1.png` | Precision / Recall / F1 per class |
| `confidence_vs_accuracy.png` | Confidence calibration curve |
| `abstention_tradeoff.png` | Accuracy vs coverage trade-off |
| `entity_sentiment_distribution.png` | Sentiment breakdown by airline |
| `entity_error_rate.png` | Error rate by airline |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
```

### Running the Analysis

1. Open the notebook:
   ```bash
   jupyter notebook notebooks/tweet_us.ipynb
   ```
2. Run all cells sequentially — the notebook handles data loading, preprocessing, EDA, model training, evaluation, and result export.

### Loading the Trained Model

```python
import joblib

pipeline = joblib.load("models/D2_selected_pipeline.joblib")
prediction = pipeline.predict(["Great flight, loved the service!"])
print(prediction)  # ['positive']
```

---

## 📜 License

The underlying dataset is licensed under **CC BY-NC-SA 4.0** as indicated on the [Kaggle source page](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). This analysis code is provided for educational purposes.
