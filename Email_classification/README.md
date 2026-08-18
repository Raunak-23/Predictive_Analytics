# Email Classification Pipeline

A comprehensive email classification system implementing multi-stage benchmarking across business intent classification (D1), binary spam detection (D2/D3), and a research extension with Bi-LSTM and LLM-assisted draft generation (D4).

---

## 📊 Project Overview

This project implements a complete email classification pipeline following the VIT Predictive Lab manual methodology. It covers three distinct classification tasks:

| Stage | Task | Dataset | Classes | Best Model | Test Macro F1 |
|-------|------|---------|---------|------------|---------------|
| **D1** | Business Intent Classification | `business_email_intent.csv` (2000 emails) | 4 (complaint, feedback, inquiry, request) | MultinomialNB (TF-IDF) | **1.000** |
| **D2** | Binary Spam Detection (Enron) | Enron-Spam (~31.6k emails) | 2 (legitimate, spam) | LinearSVC (TF-IDF) | **0.998** |
| **D3** | Binary Spam Detection (SpamAssassin) | SpamAssassin (~6k emails) | 2 (legitimate, spam) | ComplementNB (TF-IDF) | **0.995** |
| **D4** | Challenge Set Evaluation | 24 curated cases | 4 intents | MultinomialNB + LLM drafts | 100% classification accuracy |

---

## 🏗️ Architecture & Implementation Details

### Stage D1: Business Intent Classification (Multiclass)
- **Dataset**: Kaggle `yasirali646/email-intent-classification` (2000 emails, 4 classes)
- **Features**: TF-IDF (unigram+bigram, 30k max features, sublinear TF)
- **Models Compared**: Dummy, MultinomialNB, ComplementNB, LogisticRegression, LinearSVC
- **Validation**: 5-fold stratified CV on training split (80/20), locked test evaluation
- **Best Model**: MultinomialNB — perfect test performance (Macro F1 = 1.0)
- **Artifacts**: Fitted pipeline, split manifest, CV tables, confusion matrices, bootstrap 95% CI

### Stage D2/D3: Binary Spam Classification
- **D2 Source**: HuggingFace `SetFit/enron_spam` (Enron corpus, ~33k rows)
- **D3 Source**: HuggingFace `bvk/SpamAssassin-spam` (SpamAssassin corpus, ~9k rows)
- **Normalization**: Unified schema `email_id, subject, body, label, dataset_id`
- **Pipeline**: TF-IDF vectorizer inside sklearn Pipeline (leakage-safe)
- **Cross-Dataset Transfer**: Enron→SpamAssassin & SpamAssassin→Enron evaluation
- **Ablation Studies**: Subject-only, Body-only, Subject+Body input variants
- **Hyperparameter Sweeps**: α for NB, C for LR/SVC (5 values each)
- **Overfitting Check**: Train vs CV gap threshold (0.05)

### Stage D4: Research Extension
- **Sentence Embeddings**: `all-MiniLM-L6-v2` + LogisticRegression baseline (Macro F1: 0.993)
- **Bi-LSTM**: Trainable embeddings + Bidirectional LSTM (128-dim, 64 units, 0.3 dropout)
  - Training time: ~327s (20 epochs, early stopping)
  - Parameters: 145,348 | Model size: 0.55 MB
  - Test Macro F1: **1.000**
- **LLM Draft Generation**: Gemini 3.5 Flash Lite (capped at 15 calls, 20 RPM)
  - Template + structured JSON output with safety rules
  - Human evaluation worksheet (5-point Likert: relevance, faithfulness, tone, completeness, safety)
- **Challenge Set**: 24 cases across 4 intents × 3 difficulties × 2 categories

---

## 📈 Key Metrics Summary

### D1 Business Intent (Locked Test)
| Metric | Value |
|--------|-------|
| Accuracy | 1.000 |
| Macro F1 | 1.000 |
| Weighted F1 | 1.000 |
| Bootstrap 95% CI | [1.000, 1.000] |
| External Test Accuracy | 0.750 |
| External Macro F1 | 0.749 |

### Per-Class Performance (D1 Test)
| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| complaint | 102 | 1.0 | 1.0 | 1.0 |
| feedback | 101 | 1.0 | 1.0 | 1.0 |
| inquiry | 95 | 1.0 | 1.0 | 1.0 |
| request | 102 | 1.0 | 1.0 | 1.0 |

### D2/D3 Best Models (Locked Test)
| Dataset | Best Model | Test Macro F1 | 95% CI |
|---------|------------|---------------|--------|
| Enron-Spam | LinearSVC | 0.998 | [0.995, 1.000] |
| SpamAssassin | ComplementNB | 0.995 | [0.988, 1.000] |

### Representation Comparison (D1)
| Model | Representation | Macro F1 | Params | Fit Time | Model Size |
|-------|---------------|----------|--------|----------|------------|
| MultinomialNB | TF-IDF (1,2) sparse | 1.000 | ~22k | 2.1s | 4.2 MB |
| LogisticRegression | TF-IDF (1,2) sparse | 1.000 | ~22k | 3.4s | 4.2 MB |
| all-MiniLM-L6-v2 + LR | Dense embeddings (384-d) | 0.993 | 390k | 19.0s | 90 MB |
| Bi-LSTM (trainable) | Trainable emb + BiLSTM | 1.000 | 145k | 327s | 0.55 MB |

---

## 🚀 How to Run

### Prerequisites
```bash
# Python 3.10+
# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
# Copy example env and add API keys
cp .env.example .env
# Edit .env with:
# GEMINI_API_KEY=your_key_here  # For D4 LLM draft generation
# KAGGLE_USERNAME=...           # For dataset downloads
# KAGGLE_KEY=...
```

### Run Full Pipeline (Notebooks)

**Stage D1: Business Intent Classification**
```bash
jupyter notebook notebooks/email-classify.ipynb
```

**Stage D2/D3: Binary Spam Classification**
```bash
jupyter notebook notebooks/EmailClassification_D2_D3.ipynb
```

**Stage D4: Research Extension (Bi-LSTM + LLM Drafts)**
```bash
jupyter notebook notebooks/EmailClassification_D4.ipynb
```

### Run Data Preparation Scripts
```bash
# Download raw datasets from Kaggle/HuggingFace
python src/download_data.py

# Normalize D2/D3 to unified schema
python src/prepare_d2_d3.py
```

### Expected Outputs
```
outputs/
├── artifacts/          # CSVs, JSONs: audit, splits, CV, test, bootstrap, configs
├── models/             # Fitted pipelines (.joblib) and Keras models
├── figures/            # All visualizations (PNG)
├── drafts/             # D4 LLM drafts & evaluation worksheets
└── d4_draft_*.csv      # Challenge set results
```

---

## 🔬 Reproducibility

All experiments are fully reproducible:
- **Fixed random seed**: `RANDOM_STATE = 42` (numpy, sklearn, tensorflow)
- **Locked train/test splits**: Persisted in `artifacts/split_manifest*.json`
- **No test-set tuning**: Test sets evaluated exactly once after model selection
- **Leakage-safe**: TF-IDF fitted only inside CV folds via sklearn Pipeline
- **Complete artifact inventory**: See `outputs/artifacts/artifact_inventory.csv`

---

## 🏷️ Technology Tags

| Category | Technologies |
|----------|--------------|
| **Core ML** | scikit-learn 1.7+, TensorFlow/Keras 2.x |
| **NLP/Embeddings** | sentence-transformers (all-MiniLM-L6-v2), TF-IDF |
| **Data** | pandas, numpy, datasets (HuggingFace), kagglehub |
| **Visualization** | matplotlib, seaborn |
| **Persistence** | joblib, json, keras model format |
| **LLM Integration** | google-genai (Gemini API), python-dotenv |
| **Experiment Tracking** | Custom artifact system (CSV/JSON manifests) |
| **Notebooks** | Jupyter (.ipynb → .html/.pdf exports) |

---

## 📁 Project Structure

```
Email_classification/
├── data/
│   ├── business_email_intent.csv
│   ├── d2_d3_normalized/
│   │   ├── enron_spam.csv
│   │   └── spamassassin.csv
│   ├── d4_challenge_set.csv
│   └── external_test_emails.csv
├── notebooks/
│   ├── email-classify.ipynb           # D1: Business Intent
│   ├── EmailClassification_D2_D3.ipynb # D2/D3: Binary Spam
│   └── EmailClassification_D4.ipynb    # D4: Research Extension
├── outputs/
│   ├── artifacts/                      # All metric tables, configs, manifests
│   ├── models/                         # Fitted pipelines & models
│   ├── figures/                        # All plots (PNG)
│   └── drafts/                         # D4 LLM drafts
├── src/
│   ├── download_data.py               # Dataset downloaders
│   └── prepare_d2_d3.py               # D2/D3 normalization
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📝 Notes

- **D1** achieves perfect classification on the business intent task — this is expected for the curated dataset but validated via bootstrap CI and external test (75% accuracy on unseen data).
- **D2/D3** demonstrate measurable domain shift in cross-dataset transfer (Enron→SpamAssassin drop ~0.03 Macro F1), consistent with literature.
- **D4 Bi-LSTM** matches TF-IDF performance with far fewer parameters (145k vs ~22k non-zero) but requires GPU/longer training.
- **LLM drafts** are rate-limited (15 calls, 20 RPM) and include safety guardrails (placeholder enforcement, no invented facts).

---

## 📄 License

Educational/Research use. Datasets sourced from public repositories (Kaggle, HuggingFace) with their respective licenses.