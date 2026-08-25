## Interpretation of Locked-Test Results

### Overall Performance Overview
* **Selected Model:** `LinearSVC`
* **Test Accuracy:** 0.8005 (80.05%)
* **Test Macro-F1:** **0.7412**
* **Test Weighted-F1:** 0.7990

---

### Generalization & Stability Analysis
* **Validation Macro-F1:** 0.7412
* **Generalization Gap (Val - Test):** +0.0000
* **Stability Assessment:** Minimal generalization gap detected. The model demonstrates high stability and does not overfit to the validation split.

---

### Per-Class Performance Breakdown
* **Negative Sentiment:** F1-Score = **0.8759** (Majority class; benefits from dense sample presence)
* **Neutral Sentiment:** F1-Score = **0.6388** (Hardest class; frequent overlap with negative/positive boundary cases)
* **Positive Sentiment:** F1-Score = **0.7089** (Strong recall despite lower class frequency)

---

### Key Failure Modes & Error Diagnostics
1. **Neutral-Negative Confusion:** Misclassifications predominantly occur between `neutral` and `negative` classes due to implicit complaints and customer support inquiries lacking explicit sentiment keywords.
2. **Short Text & Context Constraints:** Brief tweets with heavy jargon or implicit sarcasm present the highest rate of false predictions.