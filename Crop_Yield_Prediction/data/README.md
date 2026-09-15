# Data Acquisition and Storage Guide

This document outlines the data acquisition, storage policies, and provenance governance for **Experiment 08: Agricultural Predictive Analytics**.

---

## 1. Track A: D1 Government Crop Production Statistics

- **Source:** Open Government Data (OGD) Platform India (`https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997`), published by the Ministry of Agriculture & Farmers Welfare.
- **Resource:** *District-wise, season-wise crop production statistics from 1997*.
- **Acquisition Mechanism:** Programmatically retrieved via the official REST API endpoint using `src/acquire_d1.py`.
- **API Credentials:** The API key must be supplied via an environment variable (`GOV_API_KEY`) loaded from a local `.env` file. **Never commit actual credentials to Git.**
- **Environment Setup:**
  ```bash
  # In .env (never committed)
  GOV_API_KEY=your_data_gov_in_api_key_here
  ```
- **How to Run Acquisition:**
  ```bash
  python -m src.acquire_d1
  ```
- **Raw Destination:** `data/raw/d1_government_crop_statistics.csv`
- **Canonical Destination:** `data/processed/rice_canonical.csv`
- **Exclusion Policy:** Raw datasets are strictly excluded from GitHub tracking via `.gitignore` (`data/raw/*`) because redistribution rights and licensing conditions depend on source terms and permissions.

---

## 2. Track B: D3 Crop Recommendation Dataset

- **Source:** `https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset`
- **Placement:** Placed under `data/raw/Crop_recommendation.csv` (2,200 rows, 22 balanced classes, 7 soil and weather features).
- **Processing:** Converted to canonical schema with unique identifiers via:
  ```bash
  python -m src.prepare_data
  ```
- **Canonical Destination:** `data/processed/crop_labels_canonical.csv`
