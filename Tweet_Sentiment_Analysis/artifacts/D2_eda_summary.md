# EDA Summary – D2 Dataset

## Class Distribution
- Negative: 9178 (62.7%)
- Neutral: 3099 (21.2%)
- Positive: 2363 (16.1%)

## Tweet Length (Characters)
- Mean: 103.8
- Median: 114.0
- Std: 36.3

## Top Terms per Class
- **Negative**: cancelled, delay, bad, issue, help, wait, awful, worst, refund, customer
- **Neutral**: flight, airline, time, book, change, need, question, confirm, check, schedule
- **Positive**: thanks, great, love, amazing, awesome, best, friendly, fantastic, incredible, happy

## Null/Duplicate Summary
- Null rows: 14638
- Duplicate tweet IDs: 155
- Duplicate tweet text: 213

## Key Insights
- Imbalance: negative tweets dominate; macro-F1 is the appropriate metric.
- Tweets are short; max_len 144 characters covers most tweets.
- Lexical differences are clear between classes; TF-IDF should perform well.
- Sarcasm and negation are not captured by unigrams/bigrams alone—deep learning may help.