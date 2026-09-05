# MDI3003 — Experiment 07: Recommendation System with Random Forest
## Consolidated Step-by-Step Implementation Plan

This reorganizes the manual (Sections 1–27 + Appendices A–E) into one linear build order, cross-referenced against the rubric (Sec. 25) and the failure modes called out in `Lab_quality_expectations.pptx`. The pipeline is **the same for every dataset** — only the raw field names differ — so this is a single common plan with a per-dataset schema-mapping table (Stage 1.3), not four separate pipelines.

---

## 0. Datasets and Dataset-Selection Rule (Manual Sec. 5)

| ID | Dataset | Link | Role | Size (as verified in manual) |
|---|---|---|---|---|
| **D1** | UCI Online Retail | https://archive.ics.uci.edu/dataset/352/online+retail | **Core / mandatory.** Preferred dataset for the 3-hour core pipeline. | 541,909 transactions, UK online retailer, 01-Dec-2010 to 09-Dec-2011. Fields: invoice, product code/description, quantity, invoice date, unit price, customer ID, country. |
| **D2** | Retailrocket Recommender System Dataset | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | **Advanced benchmark #1** (implicit feedback). | 2,756,101 events (views/add-to-cart/transactions) from 1,407,580 visitors; item properties + category tree. |
| **D3** | Instacart Market Basket Analysis | https://www.kaggle.com/competitions/instacart-market-basket-analysis/data | **Advanced benchmark #2** (next-basket). | Orders, products, aisles/departments, prior order-product interactions. |
| **D4** | UCI Online Retail II (optional) | https://archive.ics.uci.edu/dataset/502/online+retail+ii | **Optional cross-dataset replication.** | ~1.07M instances, 2 years of transactions. |

**Selection rule (verbatim intent of Sec. 5.1):**
- Core lab → D1, using the instructor-frozen extract, with the instructor publishing exact file/hash, date range, and train/val/test cutoff dates before the session.
- Advanced learners → replicate the protocol on **D2 or D3**. You may **not** claim direct metric comparability across datasets unless the target definition and candidate-generation policy are harmonized (same K, same relevance definition, same candidate-window logic).
- D4 is only for an optional robustness/replication extension, not for a second full pipeline.

**⚠️ PPTX-flagged failure this stage prevents:** *"Second dataset is incomplete — mentioned, but has no separate model results, plots, or interpretation."* Whichever of D2/D3 you pick for the advanced comparison must go through **every stage below in full** (its own EDA, its own baseline, its own RF, its own metrics table, its own 5-case audit) — not a one-paragraph mention. Treat it as a second complete run of this same pipeline.

Record for **each** dataset you touch (Manual Sec. 6.2 — Dataset Card): source URL, access date, version/hash, raw rows, filtered rows, unique customers/users, unique items, date range, return/cancellation policy, missing-ID policy, privacy note.

---

## 1. Environment, Setup, and Reproducibility (Manual Sec. 10.1)

### 1.1 Fixed seed and package pinning
```python
SEED = 42
np.random.seed(SEED)
```
Record `platform.platform()`, `sklearn.__version__`, and a `requirements.txt`/`environment.yml` — this is the metadata the PPTX explicitly calls out as often missing ("seed, package versions, split protocol, threshold, and tuned hyperparameters are recorded").

### 1.2 Directory layout (so every artifact has a fixed home)
```
/models/           → random_forest.joblib (+ one per advanced dataset)
/artifacts/         → feature_schema.json, split_manifest.json, candidate_policy.json
/figures/           → all *.png, one subfolder per dataset
/outputs/           → recommendations.csv, ranking_metrics.csv, candidate_recall.csv, error_analysis.csv
```

### 1.3 Per-dataset schema mapping (only place the datasets structurally differ)

| Concept | D1 (Online Retail) | D2 (Retailrocket) | D3 (Instacart) |
|---|---|---|---|
| Customer/user ID | `CustomerID` | `visitorid` | `user_id` |
| Item ID | `StockCode` | `itemid` | `product_id` |
| Transaction/order ID | `InvoiceNo` | `transactionid` | `order_id` |
| Timestamp | `InvoiceDate` | `timestamp` | `order_number` (ordinal, not clock time) |
| Quantity/value | `Quantity`, `UnitPrice` | event type (view/addtocart/transaction) | quantity implicit (1 per line) |
| Item metadata | `Description` | category tree, item properties | aisle, department |

Everything from Stage 2 onward is written once against generic column names (`user_id`, `item_id`, `order_id`, `ts`, `amount`) — rename at load time so the rest of the notebook is dataset-agnostic.

---

## 2. Data Loading and Audit (Manual Sec. 10.2, Checkpoint 0–20 min)

1. Load raw file(s), assert required columns are present (`assert not missing`).
2. Parse timestamps to proper datetime.
3. Compute and log: row count, unique users, unique items, date range, % missing user ID, % negative/zero quantity or price, % cancellations.
4. **Visualization (Manual Sec. 17, item 1):** transaction/event volume over time — line chart, title + axis labels, 2–3 sentence interpretation (seasonality, gaps, outlier spikes).

**Deliverable:** Dataset Card (Sec. 6.2) filled with real numbers — not "TBD".

---

## 3. Data Cleaning and Transaction Integrity (Manual Sec. 6.1, 10.3)

1. Drop rows with unusable/missing user ID — **log the count removed**.
2. Handle cancellations/returns explicitly (D1: `InvoiceNo` starting with `'C'`) — never count a return as a positive.
3. Filter `Quantity > 0`, `UnitPrice/price >= 0`.
4. Deduplicate exact duplicate lines while preserving legitimate repeated quantities.
5. Compute `Amount = Quantity × UnitPrice` (or equivalent).
6. Strip any direct identifiers (names, emails, payment info) if present.

**Deliverable:** a short cleaning log (rows in → rows out, by rule) — this becomes a rubric line item under "Dataset provenance & transaction integrity" (7 marks).

---

## 4. Exploratory Data Analysis (EDA)

This is the manual's Sec. 17 visualization list, run **before** modeling so cleaning/feature decisions are evidence-based, plus standard EDA additions.

### 4.1 Required visualizations (from Sec. 17)
1. Transaction volume over time *(Stage 2)*
2. Top 15 items by transaction count — bar chart
3. Customer purchase-frequency distribution — histogram
4. Customer recency/frequency/monetary (RFM) distributions — 3 histograms or a small-multiple grid
5. Class balance after negative sampling *(produced in Stage 7, but plan for it now)*
6. Feature-importance chart *(produced in Stage 11)*
7. Precision@K / Recall@K vs K *(produced in Stage 13)*
8. Popularity baseline vs Random Forest ranking metrics *(Stage 13)*
9. Recommendation-score distribution for positives vs negatives *(Stage 12)*
10. Catalog-coverage / recommendation-popularity plot *(extension, Stage 15)*

### 4.2 Additional EDA worth adding (not explicitly listed but standard practice, and defensible under "Report quality")
- Missingness heatmap / bar (before cleaning).
- Country/region distribution (D1 has a `Country` field) — informs whether region is a useful context feature.
- Price distribution (log-scale) and outlier check.
- Basket-size distribution (items per invoice/order).
- Repeat-purchase rate: fraction of (user, item) pairs bought more than once — directly informs the repeat-vs-novel policy in Stage 6.
- Sparsity of the user–item matrix (a single number: `1 − nnz/(users×items)`), since it motivates the candidate-generation design in Stage 6.

**Every figure needs:** descriptive title, labelled axes, readable scale, and **2–3 sentences of interpretation** — this is the exact "Visualizations need interpretation, not decoration" requirement from the PPTX. A caption says what the figure *is*; the paragraph says what it *means* for recommendation quality or risk.

---

## 5. Chronological Train/Validation/Test Split (Manual Sec. 11.1)

1. Freeze **exact cutoff dates** (not per-student quantiles — Manual: "instructor should freeze exact cutoff dates").
   ```python
   train_end = pd.Timestamp('YYYY-MM-DD')   # instructor-fixed
   val_end   = pd.Timestamp('YYYY-MM-DD')   # instructor-fixed
   train_hist  = df[df.ts <  train_end]
   val_future  = df[(df.ts >= train_end) & (df.ts < val_end)]
   test_future = df[df.ts >= val_end]
   ```
2. **Visualization:** a time-window diagram (a simple horizontal timeline bar showing train/val/test regions) — required evidence per Checkpoint 2 (20–40 min).
3. Acceptance tests (Appendix C) to run immediately and keep in the notebook:
   ```python
   assert train_hist.ts.max() < train_end
   assert val_future.ts.min() >= train_end
   assert test_future.ts.min() >= val_end
   assert set(test_future.index).isdisjoint(set(train_hist.index))
   ```

**This is the single highest-weighted rubric line (10 marks: "Temporal split & leakage prevention")** — do not skip the diagram or the assertions.

---

## 6. Candidate Generation and Negative Sampling (Manual Sec. 11.2–11.4)

1. Define the eligible candidate universe from `train_hist` only: items with ≥ M buyers or in the top-N most frequent, **bounded to ~500–2,000 items** (instructor-approved range; assert this bound in code, Appendix C).
2. Decide and document the **repeat-purchase policy**: are already-purchased items eligible again? (Matters a lot for D1/grocery-like data.)
3. **Candidate-recall audit (Sec. 11.3) — do this before trusting any ranking metric:**
   ```
   Candidate Recall = |future-relevant items ∩ candidate set| / |future-relevant items|
   ```
   Report this as its own number, and the % of evaluation users for whom *all* future positives are representable. A low candidate recall caps the maximum possible Recall@K regardless of how good the model is — report it, don't silently drop unretrievable positives.
4. Negative sampling: reproducible, seeded, from the candidate pool minus positives (Sec. 11.4 `sample_negatives`).
5. **Visualization:** class balance after negative sampling (bar chart of positive vs sampled-negative counts).

**Deliverable:** `candidate_policy.json` (saved artifact) + `Candidate_Recall.csv`.

---

## 7. Feature Engineering (Manual Sec. 12, 8.1)

Build three feature blocks, **all computed strictly from history before the cutoff** (leakage rule, Sec. 12 "Leakage rule" box):

| Level | Features |
|---|---|
| Customer | txn count, unique items, recency (days since last), frequency, monetary total, avg basket size, active days, preferred category |
| Item | txn count, unique buyers, recent popularity, price stats, category, repeat-purchase rate |
| Customer–item (pair) | prior purchase count, days since last purchase of this item, share of customer spend on this item, repeat indicator, co-purchase frequency |
| Context (optional) | day/month/season, country/region |

Join these onto the positive + sampled-negative pairs from Stage 6 to form `X_train`, `y_train`, `X_val`, `y_val`.

**Deliverable:** a **feature dictionary** (table: feature name → definition → computed-from) — this is explicit required evidence at Checkpoint 3 (60–95 min) and a separate rubric line (12 marks).

**Acceptance checks (Appendix C):**
```python
assert not X_train[feature_cols].isna().any().any()
assert set(np.unique(y_train)).issubset({0,1})
```

---

## 8. Baseline Model: Popularity (Manual Sec. 13.1, Experiment E1)

```python
popular_items = (train_hist.groupby('item_id')['order_id']
                  .nunique().sort_values(ascending=False).index.tolist())
def popularity_recommend(seen_items, k=10):
    return [i for i in popular_items if i not in seen_items][:k]
```
Run this Top-K on the **same locked test customers** you'll later use for RF — this baseline is mandatory (6 marks) and is what every later comparison plot is measured against.

---

## 9. Random Forest Model Building (Manual Sec. 13.2, 8.2)

```python
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced_subsample',
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train[feature_cols], y_train)
```
`score(u,i) = rf.predict_proba(x)[:,1]` is the ranking score; candidates are sorted descending by this score.

---

## 10. Validation-Based Tuning ("CV" for this problem)

**Important correction to keep in mind:** standard k-fold CV is *not* appropriate here — it would shuffle future transactions into training folds and violate the chronological-leakage rule (Sec. 22 "Common Mistakes" table, row 1). The manual's substitute for CV is:

1. Fixed train/validation/test **windows** (already built, Stage 5) — validation is a proper held-out future period, not a random fold.
2. Grid over the hyperparameters in Sec. 13.3:

| Parameter | Core values to compare |
|---|---|
| `n_estimators` | 200, 400 |
| `max_depth` | None, 12, 20 |
| `min_samples_leaf` | 1, 2, 5 |
| `max_features` | sqrt, 0.5 |
| `class_weight` | balanced_subsample, or a justified alternative |

3. **Selection rule:** choose hyperparameters by **validation Recall@K or NDCG@K**, with PR-AUC only as a secondary diagnostic. **Never select using locked-test results** — that is explicitly the evaluation-fairness rule the PPTX calls "keep evaluation separate from development and tuning."
4. Keep a **validation table** (all configs tried, their validation metrics, the one selected and why) — required evidence at Checkpoint 5 (95–125 min) and this is exactly the kind of table the PPTX flags when it says *"tables contain 'Computed', 'Lowest', or 'Improved' instead of actual values."* Every cell must be a real number.

If you want a closer analogue to CV for extra rigor: use **rolling-origin (time-series) validation** — multiple train/val cutoffs shifted forward in time, each respecting chronology — and report mean ± std across those windows. This is optional for the core but strengthens the advanced comparison.

---

## 11. Hyperparameter Tuning — Outputs

1. Refit the **selected** configuration on the allowed development history.
2. **Visualization:** feature-importance chart (Sec. 17 item 6) — bar chart of `rf.feature_importances_`, sorted, with interpretation: which feature families (recency? pair history? popularity?) actually drive the model, and an explicit caveat that importance is model-dependence, not causality (Sec. 21 viva point).

---

## 12. Final Model — Save, Reload, Inference (Manual Sec. 14, Appendix A step 7–8)

1. Evaluate **once** on the locked test window, same candidate policy as validation.
2. Save the model:
   ```python
   joblib.dump(rf, 'models/random_forest.joblib')
   ```
3. **Reload test** — load the saved model in a fresh session and confirm predictions match the in-memory model on a verification sample (Appendix C, last assertion). This is a named final-checklist item and a PPTX-flagged gap ("saved outputs... and the selected model are present").
4. Inference function for any user:
   ```python
   def recommend_for_user(user_id, candidate_df, model, feature_cols, k=10):
       u = candidate_df[candidate_df.user_id == str(user_id)].copy()
       u['score'] = model.predict_proba(u[feature_cols])[:,1]
       return u.sort_values('score', ascending=False).head(k)
   ```
5. **Visualization:** recommendation-score distribution for positives vs negatives (Sec. 17 item 9) — overlapping histograms/density plot; interpretation should discuss separability.

---

## 13. Evaluation Metrics (Manual Sec. 15)

Compute for locked test customers, at **K = 5, 10, 20** (Experiment E3):

| Metric | Required? |
|---|---|
| Precision@K | Core |
| Recall@K | Core |
| Hit Rate@K | Core |
| Candidate Recall | Core (reported separately, diagnoses retrieval loss) |
| MAP@K | Advanced |
| NDCG@K | Advanced — primary advanced ranking metric |
| ROC-AUC | Diagnostic only (can look good under imbalance — don't lead with it) |
| PR-AUC | Diagnostic (more informative under imbalance) |
| Coverage | Extension |

**Visualizations:**
- Precision@K / Recall@K vs K (Sec. 17 item 7) — line chart, K on x-axis.
- Popularity vs Random Forest ranking metrics (Sec. 17 item 8) — grouped bar chart, this is the headline comparison the whole lab is building toward.

**Result table (Sec. 19 template) — fill every cell with a real number:**

| Model | P@5 | R@5 | HR@5 | P@10 | R@10 | HR@10 | PR-AUC |
|---|---|---|---|---|---|---|---|
| Popularity | | | | | | | |
| Random Forest | | | | | | | |
| Advanced model | | | | | | | |

This is the exact table the PPTX warns about — never leave a cell as "improved" or "best"; every number must be independently checkable against `Ranking_Metrics.csv`.

---

## 14. Error, Cold-Start, Leakage, and Bias Analysis (Manual Sec. 16)

### 14.1 Required five-case audit (verbatim from Sec. 16.1) — with real user IDs and real numbers, not descriptions:
1. A successful personalized recommendation, with supporting history shown.
2. A relevant item **missed by RF but caught by popularity baseline**.
3. A case where **RF beats popularity**.
4. A sparse/cold-start customer.
5. A questionable recommendation traceable to popularity dominance, negative-sampling artifact, or a feature limitation.

Each case = user ID, history snippet, true future item(s), Top-5 list, hit/miss, one sentence of comment (Sec. 19 template table).

### 14.2 Systematic checks (Sec. 16 table)
- Popularity domination: does RF just recommend the same few items to everyone? (compare against the coverage plot, Stage 15)
- Cold-start user/item handling and fallback policy.
- Repeat-purchase bias: is discovery being crowded out by consumables?
- Re-verify: no temporal leakage, no return/cancellation contamination.

**Deliverable:** `Error_Analysis.csv` + a short written section — this is where the PPTX's "interpretation is too thin" failure shows up most; each case needs its explanation sentence, not just the table row.

---

## 15. Advanced Extensions (only if pursuing the advanced pathway — Sec. 18, 18.1)

Run **the same Stages 4–14** against **D2 or D3** as the second dataset, plus:

1. **Mandatory advanced benchmark:** at least one of item-item Collaborative Filtering or Matrix Factorization/ALS, compared against RF on the **same split, same target definition, same candidate universe** (Sec. 18.1, Appendix D).
2. Optional further extensions: LightFM, Neural Collaborative Filtering, Two-Tower retrieval.
3. **Uncertainty requirement:** for any stochastic/latent model, report either ≥3 random seeds or a user-level bootstrap 95% CI for the principal ranking metric (Recall@K, NDCG@K).
4. **Visualization:** catalog-coverage / recommendation-popularity plot (Sec. 17 item 10).
5. Efficiency table (Sec. 19 last template): training time, scoring latency/user, model size, catalog coverage, complexity note — for Popularity, RF, and the CF/MF/Neural model.
6. Close with the **advanced comparison question** explicitly answered in prose: does the RF gain over popularity justify its engineering cost? Does the latent/neural model's gain justify its added complexity?

**This entire section is what "second dataset is incomplete" means in practice — it needs its own full metrics table, its own 5-case audit, its own plots, not a paragraph.**

---

## 16. Responsible Recommendation and Privacy (Manual Sec. 23)

Write a short section covering: data minimization, no exposure of identifiable histories in screenshots/report, popularity-bias/concentration assessment, explicit statement that the system must not be used for discriminatory pricing, protected-class targeting, or credit decisions, and disclosure of any external code/AI/dataset assistance used.

---

## 17. Reproducibility and Artifacts — Full Checklist (Manual Sec. 24, 26 + PPTX Slide 6)

**Saved artifacts (per dataset):**
- `models/random_forest.joblib` (+ one per advanced dataset/model)
- `artifacts/feature_schema.json`
- `artifacts/split_manifest.json` (exact cutoff dates used)
- `artifacts/candidate_policy.json`
- `outputs/Ranking_Metrics.csv`
- `outputs/Candidate_Recall.csv`
- `outputs/Recommendations.csv`
- `outputs/Error_Analysis.csv`
- `outputs/Advanced_Uncertainty.csv` (advanced learners)
- `figures/*.png` (every plot listed in Stage 4.1, titled and captioned)
- `README.md` — how to run the notebook top-to-bottom in a clean runtime, plus environment/versions/seed

**Metadata to record explicitly (this is the PPTX's #1 and #6 flagged gaps):**
- Random seed, package versions, exact split dates, candidate-policy version, selected hyperparameters, decision threshold (if one is used anywhere).

**Final sanity pass before submission (merge of Manual Sec. 26 + PPTX Slide 8):**
- [ ] Every table cell is a real, checkable number — no "Computed", "Best", "[insert]".
- [ ] The second/advanced dataset has full model results, plots, and interpretation — not a summary paragraph.
- [ ] Notebook/script runs top-to-bottom in a clean runtime; GitHub link (if used) is live and matches the code shown.
- [ ] Visual evidence (confusion-matrix-equivalent: Precision/Recall@K plots, score distributions, feature importance) exists for **every** dataset used, not just the core one.
- [ ] Every figure and table has a 2–3 sentence interpretation underneath it.
- [ ] No leftover template placeholders, unrelated sections, or duplicate content.
- [ ] Report, code outputs, and CSVs tell the same story (numbers match across all three).
- [ ] Model saved and successfully reloaded with matching predictions.
- [ ] Names/registration numbers/filenames are consistent across every submitted file.

---

## 18. Submission Package (Manual Sec. 24)

```
RegistrationNumber_Lab07_Recommender_RF.ipynb
RegistrationNumber_Lab07_Report.pdf
RegistrationNumber_Lab07_Ranking_Metrics.csv
RegistrationNumber_Lab07_Candidate_Recall.csv
RegistrationNumber_Lab07_Recommendations.csv
RegistrationNumber_Lab07_Error_Analysis.csv
RegistrationNumber_Lab07_Advanced_Uncertainty.csv   (advanced learners)
RegistrationNumber_Lab07_README.md
models/random_forest.joblib
artifacts/feature_schema.json
artifacts/split_manifest.json
artifacts/candidate_policy.json
figures/*.png
```

---

## 19. Stage → Rubric Cross-Reference (100 marks, Sec. 25)

| Stage(s) above | Rubric criterion | Marks |
|---|---|---|
| 0 | Problem formulation & recommendation boundary | 6 |
| 0, 2, 3 | Dataset provenance & transaction integrity | 7 |
| 5 | Temporal split & leakage prevention | 10 |
| 8 | Popularity baseline | 6 |
| 6 | Candidate generation & negative sampling | 10 |
| 7 | Feature engineering | 12 |
| 9–11 | Random Forest implementation & validation | 14 |
| 12 | Top-K ranking & recommendation logic | 10 |
| 13, 4 | Evaluation & visualizations | 10 |
| 14 | Error/cold-start/bias analysis | 7 |
| 16 | Responsible recommendation | 4 |
| 17 | Reproducibility & artifacts | 3 |
| all | Report quality | 1 |
