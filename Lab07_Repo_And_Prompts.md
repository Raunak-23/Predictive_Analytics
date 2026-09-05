# Repo Structure + Stage-Wise AI Prompts

## 1. Project Repo Structure

```
recommender-lab07/
├── data/
│   ├── raw/
│   │   ├── d1_online_retail/
│   │   └── d2_advanced/                 # retailrocket or instacart, whichever you pick
│   └── processed/
│       ├── d1_online_retail/
│       └── d2_advanced/
├── notebooks/
│   ├── 01_d1_online_retail.ipynb
│   └── 02_d2_advanced.ipynb
├── scripts/
│   ├── run_d1_online_retail.py          # .py version of notebook 1 (run first)
│   └── run_d2_advanced.py               # .py version of notebook 2
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── splitting.py
│   ├── candidates.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── artifacts.py
├── artifacts/
│   ├── d1_online_retail/
│   └── d2_advanced/
├── models/
│   ├── d1_online_retail/
│   └── d2_advanced/
├── results/
│   ├── d1_online_retail/
│   └── d2_advanced/
├── figures/
│   ├── d1_online_retail/
│   └── d2_advanced/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 2. Standing Instructions (give this once, before Stage 1, as project-level context)

```
PROJECT CONTEXT — read before every stage:
Building a Random-Forest-based recommender per the attached lab manual, for two
datasets (D1 = UCI Online Retail = core, D2 = Retailrocket or Instacart = advanced
benchmark). Repo layout is fixed — [paste the tree above]. Work stage by stage,
one prompt per stage; do not jump ahead.

CODE QUALITY STANDARD — apply to every file you write:
- All reusable logic goes in src/ as plain functions — never duplicate logic
  inside scripts/ or notebooks/. scripts/ and notebooks/ only call src functions.
- Every function: type hints, one-line docstring, no hardcoded column names —
  accept user_col, item_col, order_col, ts_col, amount_col as parameters (with
  sensible defaults) so the same function works on both datasets.
- No hardcoded file paths, dates, or magic numbers — read from src/config.py or
  function arguments.
- Deterministic: every stochastic step takes a seed parameter, default from config.
- No bare except; raise/assert on invalid input instead of silently continuing.
- PEP8, black-formatted, no unused imports, no commented-out dead code.
- Every function that produces a metric, table, or plot returns the underlying
  data (DataFrame/dict), not just a printed string — so notebooks can build
  interpretation text FROM the real returned numbers, never as static prose.
- Stop and ask me if a stage's requirement is ambiguous — do not guess and
  silently proceed.
```

---

## 3. Stage-Wise Prompts

### Stage 1 — Config + Data Loading
```
Create src/config.py: SEED, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR,
RESULTS_DIR, FIGURES_DIR, and a DATASET_SCHEMAS dict mapping each dataset name
to its raw column names (user/item/order/timestamp/amount).

Create src/io_utils.py with load_transactions(path, schema) -> DataFrame that:
- loads the raw file, renames columns to generic names (user_id, item_id,
  order_id, ts, amount) using the schema dict, parses ts to datetime.
- asserts required columns exist; raises a clear error naming what's missing.
- returns a short audit dict alongside the DataFrame (row count, unique users,
  unique items, date range, % missing user_id) — do not print inside the
  function, just return the dict so the notebook decides how to show it.

Pay attention to: D2/D3 have different raw schemas — the function must be
schema-driven, not hardcoded to D1's column names.
```

### Stage 2 — Cleaning
```
Create src/cleaning.py with clean_transactions(df, schema, dataset_name) that:
- removes rows with missing user_id, logs count removed
- handles cancellations/returns per dataset (D1: order_id starting with 'C';
  for D2/D3 use the schema's equivalent rule — ask me if unclear for D3)
- filters non-positive quantity/price, dedups exact duplicate rows
- computes amount column if not already present
- returns (cleaned_df, cleaning_log_dict) — log must have real before/after
  counts per rule, not just a boolean.

Pay attention to: never drop rows silently — every removal rule must show up
in cleaning_log_dict with a count.
```

### Stage 3 — Universal EDA Module
```
Create src/eda.py with plotting + stats functions usable on ANY dataset:
- txn_volume_over_time(df, ts_col) -> (fig, summary_dict)
- top_n_items(df, item_col, order_col, n=15) -> (fig, DataFrame)
- purchase_frequency_distribution(df, user_col) -> (fig, summary_stats)
- rfm_distributions(df, user_col, ts_col, amount_col, cutoff) -> (fig, DataFrame)
- sparsity(df, user_col, item_col) -> float

Each function: matplotlib fig + the underlying computed data/stats returned
together — never a function that only prints. Titles/axis labels required on
every figure. Do not put interpretation text inside src/ — interpretation is
written later in the notebook FROM the returned stats.

Pay attention to: these functions must not filter or split the data — they
just describe whatever DataFrame is passed in.
```

### Stage 4 — Chronological Split
```
Create src/splitting.py with:
- chronological_split(df, ts_col, train_end, val_end) -> (train_hist, val_future,
  test_future) plus assertion checks (train max ts < train_end, etc. — mirror
  the manual's Appendix C assertions) raised as AssertionError with clear
  messages, not silent.
- split_timeline_plot(train_hist, val_future, test_future, ts_col) -> fig
  (simple horizontal bar showing the three windows).

Pay attention to: train_end/val_end are REQUIRED arguments with no default —
force the caller (notebook) to make an explicit, documented choice.
```

### Stage 5 — Candidate Generation & Negative Sampling
```
Create src/candidates.py with:
- build_candidate_universe(train_hist, item_col, order_col, min_buyers=None,
  top_n=None) -> list[item_id], enforcing the 500–2000 size bound (assert it,
  raise if outside range with the actual count in the message).
- candidate_recall(test_future, candidates, user_col, item_col) -> 
  (recall_float, pct_users_fully_covered)
- sample_negatives(user_id, positives, candidate_items, n_neg, seed) ->
  list[item_id], reproducible via seed.

Pay attention to: candidate_recall must be computed and returned as a plain
number — this is a required reported metric, not just an internal check.
```

### Stage 6 — Feature Engineering
```
Create src/features.py with:
- customer_features(hist, cutoff, user_col, order_col, item_col, amount_col,
  ts_col) -> DataFrame
- item_features(hist, cutoff, item_col, order_col, user_col, amount_col, ts_col)
  -> DataFrame
- pair_features(hist, user_col, item_col, order_col, amount_col, ts_col) ->
  DataFrame
- build_feature_matrix(pairs_df, customer_feats, item_feats, pair_feats) ->
  (X, feature_cols)
- feature_dictionary(feature_cols) -> DataFrame (name, description, source
  level) — descriptions come from a fixed lookup dict you define alongside
  the feature-building code, not invented per call.

Pay attention to: every aggregate must be computed ONLY from `hist` (data
strictly before `cutoff`) — this is the single most important leakage rule
in the whole project. Add an assertion inside build_feature_matrix that no
NaNs remain in feature_cols.
```

### Stage 7 — Models (baseline + Random Forest, reusable across datasets)
```
Create src/models.py with:
- popularity_baseline(train_hist, item_col, order_col) -> ranked list[item_id]
- popularity_recommend(ranked_items, seen_items, k) -> list[item_id]
- train_random_forest(X_train, y_train, **rf_params) -> fitted model
- score_candidates(model, X) -> np.array of probabilities
- recommend_top_k(scored_df, user_col, score_col, k) -> DataFrame of top-k
  per user

Keep this file model-agnostic where possible: train_random_forest should be
easy to swap for train_model(model_class, X, y, **params) if we add CF/MF later
— but don't over-engineer now, just keep the signature clean.

Pay attention to: no hardcoded hyperparameters inside the function bodies —
they come in as **params from the notebook/config, with defaults matching
the manual's Sec. 13.2 values.
```

### Stage 8 — Validation-Based Hyperparameter Tuning
```
Add to src/models.py: tune_random_forest(X_train, y_train, X_val, val_pairs_df,
param_grid, k=10, metric='recall') -> (best_params, results_table_df).
It must loop the grid, score validation candidates, rank by the recommendation
metric (not raw accuracy), and return a full results table (every config +
its validation score) — not just the winner.

Pay attention to: selection uses ONLY validation data. Test data must not be
passed into this function at all — enforce this by simply not accepting a
test argument.
```

### Stage 9 — Evaluation Metrics
```
Create src/evaluation.py with:
- precision_at_k, recall_at_k, hit_rate_at_k, average_precision_at_k,
  ndcg_at_k (per-user functions, pure, no I/O)
- evaluate_recommendations(recs_by_user, relevant_by_user, k_values=[5,10,20])
  -> DataFrame (one row per model, one column per metric per k) — this is the
  function both baseline and RF results feed into, so the result table is
  always built the same way for every model.

Pay attention to: recall_at_k must return NaN (not 0) for users with zero
relevant items, and evaluate_recommendations must handle that NaN correctly
when averaging (nanmean).
```

### Stage 10 — Visualization (metrics + diagnostics)
```
Add to src/visualization.py:
- metric_vs_k_plot(metrics_df, k_values, metric_name) -> fig
- model_comparison_bar(metrics_df, metric_name, k) -> fig
- score_distribution_plot(scored_df, label_col, score_col) -> fig
- feature_importance_plot(model, feature_cols) -> (fig, importances_df)
- class_balance_plot(y) -> fig

Every function returns (fig, underlying_data) so notebooks write
interpretation from underlying_data, never from eyeballing the plot.
```

### Stage 11 — Orchestration Script (per dataset)
```
Create scripts/run_d1_online_retail.py that imports ONLY from src/ and runs the
full pipeline end to end: load → clean → EDA → split → candidates → features →
baseline → RF train → tune → evaluate → save artifacts. Save every figure to
figures/d1_online_retail/, every table to results/d1_online_retail/, model to
models/d1_online_retail/, JSON artifacts to artifacts/d1_online_retail/.

No plotting/printing logic should live in this script beyond calling src
functions and saving their outputs — the script is just orchestration.

Run it, fix any errors, and confirm all expected output files exist before
telling me it's done.
```
*(Repeat the same prompt for `run_d2_advanced.py`, swapping the dataset name/paths.)*

### Stage 12 — Error / Cold-Start / Bias Audit
```
Add to src/evaluation.py: build_five_case_audit(scored_df, test_future,
train_hist, user_col, item_col, k=5) -> DataFrame with exactly the 5 required
case types from the manual (successful hit, RF miss/popularity hit, RF beats
popularity, cold-start user, questionable case) — pick ONE real qualifying
user per case type from the actual data, not synthetic examples. Return their
real IDs, history snippet, true future items, top-5 list, hit/miss flag.

Pay attention to: if a qualifying case genuinely doesn't exist in this dataset
(e.g. no cold-start users), return that row with a note explaining why, don't
fabricate one.
```

### Stage 13 — Artifacts & Reproducibility Save
```
Create src/artifacts.py with save_run_artifacts(model, feature_cols,
split_dates, candidate_policy, out_dir) that writes feature_schema.json,
split_manifest.json, candidate_policy.json, and the joblib model, then
reload_and_verify(model_path, X_sample) -> bool that reloads the model and
checks predictions match the in-memory model on a sample. Wire this into both
run scripts as the final step.
```

### Stage 14 — Convert Script to Notebook
```
Take the working, tested scripts/run_d1_online_retail.py and turn it into
notebooks/01_d1_online_retail.ipynb: same calls to src functions, but split
into logical cells with markdown cells before each stage explaining what
it does and why (short, matching the manual's stage names).

After each results-producing cell (EDA stat, metric table, plot, feature
importance, 5-case audit), add a markdown interpretation cell that is
generated by an f-string/template pulling the ACTUAL returned numbers from
that cell's output — never write static interpretation prose. If a number
changes on rerun, the interpretation text must change with it.

Do not add any new logic in the notebook — only src function calls, cell
organization, and markdown. Confirm the notebook runs top-to-bottom clean
before telling me it's done.
```
*(Repeat for notebook 2 once script 2 is confirmed working.)*

### Stage 15 — Advanced Benchmark (D2/D3 notebook only)
```
Add train_cf_baseline(...) or train_mf_baseline(...) to src/models.py (item-item
CF or ALS, your call — tell me which and why before implementing), using the
SAME candidate universe and split as the RF run on this dataset. Extend
evaluate_recommendations usage to include this model in the comparison table.
For any stochastic component, run 3 seeds and report mean ± std in the
results table, not just one run.
```
