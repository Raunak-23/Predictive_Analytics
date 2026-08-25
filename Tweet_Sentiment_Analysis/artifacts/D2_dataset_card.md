# Dataset Card: Twitter US Airline Sentiment

**Canonical source:** https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
**Version/Access date:** 2026-08-25
**License/usage terms:** CC BY-NC-SA 4.0 (as indicated on Kaggle page)
**Number of rows:** 14640
**Label distribution:** {'negative': 9178, 'neutral': 3099, 'positive': 2363}
**Train/validation/test split:** To be defined (stratified split with 70/15/15 or 80/10/10).
**Duplicate policy:** We will remove duplicate tweet_ids (if any) and consider dropping near-duplicate text after normalisation if they are exact copies.
**Language:** English
**Domain:** Tweets about US airlines (customer service, delays, flights)
**Annotation method:** Human annotation via Crowdflower
**Known sampling limitations:** Data collected in February 2015; only tweets about six US airlines; may not represent all customers or current sentiment.

**Fields excluded for privacy/leakage:**
- `airline_sentiment_confidence`: Annotation confidence – target leakage
- `negativereason`: Post‑label reason – leakage
- `negativereason_confidence`: Confidence on reason – leakage
- `airline_sentiment_gold`: Gold label – leakage
- `negativereason_gold`: Gold reason – leakage
- `name`: Username – PII
- `tweet_coord`: Coordinates – privacy/spurious correlation
- `tweet_created`: Timestamp – not used as predictor
- `tweet_location`: Location – privacy
- `user_timezone`: Timezone – not predictive