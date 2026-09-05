# MDI3003: Advanced Predictive Analytics — Experiment 07
## Transaction Recommender System using Supervised Random Forest

**Student Registration Number:** 23MID0045  
**Course Code:** MDI3003 (Advanced Predictive Analytics)  
**Semester:** Fall Semester 2026-2027  
**Faculty:** Dr. Durgesh Kumar  
**Target Datasets:**
- Dataset D1: UCI Online Retail (Giftware & Consumables, 541,909 raw rows)
- Dataset D3: Instacart Market Basket Analysis (Grocery Replenishment, 156,227 multi-order rows)

---

### 1. Repository Structure & Deliverables

```
Transaction_Recommender/
├── 23MID0045_Lab07_Report.docx                  <- Comprehensive Publication-Grade Laboratory Report (Word)
├── 23MID0045_Lab07_Report.pdf                   <- Report (PDF format if converted)
├── 23MID0045_Lab07_Recommender_RF.ipynb         <- Executable Jupyter Notebook (D1 Core Benchmark)
├── 23MID0045_Lab07_Ranking_Metrics.csv          <- Final Metric Table (Section 19 Template, P@K, R@K, HR@K, NDCG@K)
├── 23MID0045_Lab07_Candidate_Recall.csv         <- Candidate Bounding & Candidate Recall Analysis
├── 23MID0045_Lab07_Recommendations.csv          <- Top-K Recommendation Rankings for Evaluation Cohort
├── 23MID0045_Lab07_Error_Analysis.csv           <- Five-Case Real Customer Error and Qualitative Audit
├── 23MID0045_Lab07_Advanced_Uncertainty.csv     <- Multi-Seed Uncertainty Report (3 Seeds: 42, 101, 2024)
├── 23MID0045_Lab07_README.md                    <- Submission Guide & Verification Log
├── src/                                         <- Modular Library Architecture
│   ├── config.py                                <- Reproducible Seeds & Schemas
│   ├── io_utils.py                              <- Schema-Driven Ingestion
│   ├── cleaning.py                              <- Hygiene, Returns & Deduplication
│   ├── eda.py                                   <- Publication-Grade EDA Visualizations
│   ├── splitting.py                             <- Chronological Splits & Leakage Assertions
│   ├── candidates.py                            <- Bounded Catalog & Negative Sampling
│   ├── features.py                              <- 20 RFM & Customer-Item Features
│   ├── models.py                                <- Baseline, Random Forest & TruncatedSVD MF
│   ├── evaluation.py                            <- Ranking Metrics & 5-Case Qualitative Audit
│   ├── visualization.py                         <- Diagnostic Curves, Coverage & Lorenz Plots
│   └── artifacts.py                             <- Serializer & Reload Equivalence Checker
├── scripts/
│   ├── run_d1_online_retail.py                  <- End-to-End Orchestrator for D1
│   ├── run_d2_advanced.py                       <- End-to-End Orchestrator for D3
│   ├── build_notebooks.py                       <- Generates 01_d1_online_retail & 02_d2_advanced notebooks
│   ├── generate_report.py                       <- Builds 23MID0045_Lab07_Report.docx
│   └── prepare_submission.py                    <- Verifies and packages deliverables
├── notebooks/
│   ├── 01_d1_online_retail.ipynb                <- Core Random Forest Recommender
│   └── 02_d2_advanced.ipynb                     <- Advanced Instacart Benchmark & Multi-Seed MF
├── models/
│   ├── d1_online_retail/random_forest.joblib    <- Serialized D1 Model Checkpoint
│   └── d2_advanced/random_forest.joblib         <- Serialized D3 Model Checkpoint
├── artifacts/
│   ├── d1_online_retail/                        <- Feature Schema, Split Manifest, Policy JSONs
│   └── d2_advanced/                             <- D3 Checkpoints & Policies
├── figures/                                     <- High-Resolution Diagnostic PNGs (D1 & D2)
└── submission/                                  <- Self-Contained Section 24 Submission Directory
```

---

### 2. Instructions to Run and Reproduce

To reproduce the exact numerical results and figures from source:

1. **Activate Python Environment and Dependencies**:
   ```powershell
   cd "Transaction_Recommender"
   pip install -r requirements.txt
   ```

2. **Execute Dataset D1 Pipeline (UCI Online Retail)**:
   ```powershell
   python scripts/run_d1_online_retail.py
   ```
   *Execution time: ~110 seconds.* Generates all D1 CSVs, figures, and serialized model.

3. **Execute Dataset D3 Pipeline (Instacart Multi-Model Benchmark)**:
   ```powershell
   python scripts/run_d2_advanced.py
   ```
   *Execution time: ~72 seconds.* Executes Popularity, Random Forest, and Matrix Factorization across 3 random seeds (`[42, 101, 2024]`), generating multi-seed uncertainty and computational efficiency tables.

4. **Regenerate Laboratory Report**:
   ```powershell
   python scripts/generate_report.py
   ```
   Outputs the publication-ready Word document `23MID0045_Lab07_Report.docx`.

---

### 3. Key Findings & Empirical Results

- **Candidate Universe**: Top 1,000 products by historical order frequency (within instructor bounds $[500, 2000]$).
  - D1 Candidate Recall: **56.84%**
  - D3 Candidate Recall: **47.36%**
- **Headline Benchmark (Dataset D1)**:
  - **Popularity**: P@5 = 0.1016, R@5 = 0.0278, HR@5 = 0.3500, NDCG@10 = 0.1034, P@20 = 0.0756, R@20 = 0.0836, HR@20 = 0.6100.
  - **Random Forest**: P@5 = 0.2460, R@5 = 0.0952, HR@5 = 0.5900, NDCG@10 = 0.2643, P@20 = 0.1877, R@20 = 0.2349, HR@20 = 0.7920.
  - **Relative Gain**: **+142% P@5**, **+242% R@5**, **+155% NDCG@10**.
- **Multi-Model Benchmark (Dataset D3 Instacart Grocery)**:
  - **Popularity**: P@5 = 0.3832, R@5 = 0.0593, HR@5 = 0.7009, NDCG@10 = 0.3705, P@20 = 0.2486, R@20 = 0.1465, HR@20 = 0.8879.
  - **Random Forest**: P@5 = 0.6093, R@5 = 0.1458, HR@5 = 0.9439, NDCG@10 = 0.6401, P@20 = 0.4762, R@20 = 0.3925, HR@20 = 1.0000.
  - **Matrix Factorization (SVD)**: P@5 = 0.6224, R@5 = 0.1220, HR@5 = 0.9533, NDCG@10 = 0.5943, P@20 = 0.4145, R@20 = 0.2886, HR@20 = 0.9907.
  - **Multi-Seed Uncertainty (MF across seeds 42, 101, 2024)**:
    - Precision@10: $0.5280 \pm 0.0050$
    - Recall@10: $0.1958 \pm 0.0031$
    - HitRate@10: $0.9657 \pm 0.0044$
    - NDCG@10: $0.5943 \pm 0.0011$
- **Model Reload Verification**: Confirmed identical predictions ($0.0$ difference) between in-memory trained model and reloaded joblib artifact.
