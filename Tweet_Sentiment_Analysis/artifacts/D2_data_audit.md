# Data Audit Summary – D2 (Twitter US Airline Sentiment)

- **Total rows:** 14640
- **Total columns:** 16
- **Columns:** tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, user_timezone, text_len

## Missing Values
- `tweet_id`: 0 missing
- `airline_sentiment`: 0 missing
- `airline_sentiment_confidence`: 0 missing
- `negativereason`: 5462 missing
- `negativereason_confidence`: 4118 missing
- `airline`: 0 missing
- `airline_sentiment_gold`: 14600 missing
- `name`: 0 missing
- `negativereason_gold`: 14608 missing
- `retweet_count`: 0 missing
- `text`: 0 missing
- `tweet_coord`: 13621 missing
- `tweet_created`: 0 missing
- `tweet_location`: 4733 missing
- `user_timezone`: 4820 missing
- `text_len`: 0 missing

## Duplicate tweet IDs: 155
## Duplicate raw tweet text: 213

## Label distribution
- `negative`: 9178
- `neutral`: 3099
- `positive`: 2363

## Text length statistics
- Mean: 103.82
- Std: 36.28
- Min: 12
- Max: 186

## Entity (airline) distribution
- `United`: 3822
- `US Airways`: 2913
- `American`: 2759
- `Southwest`: 2420
- `Delta`: 2222
- `Virgin America`: 504

## Notes
- All leakage and PII fields have been identified and will be excluded from modelling (see dataset card).
- The dataset is imbalanced; macro F1 will be used as primary metric.
- Sample is from February 2015, only US airlines.