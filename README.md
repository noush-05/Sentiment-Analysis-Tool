# Sentiment Analysis Tool
This is a supervised machine learning model for text classification, as the model learns from labelled examples.

## Results 
| Version | Model           | Accuracy | Neg Recall | Pos Recall | Comments   |
|---------|-----------------|----------|------------|------------|------------|
| 01      | NB (unigrams)   | 0.7925   | 0.85       | 0.74       | too pessimistic
| 02      | NB (uni+bi)     | 0.8      | 0.84       | 0.76       | Slight improvement in positive recall but best to switch models

## Most “positive” and “negative” phrases according to version 03
Top positive n-grams:
- great
- life
- war
- family
- excellent
- truman
- best
- perfect
- mulan
- world
- overall
- perfectly
- jackie
- performance
- terrific
Top negative n-grams:
- bad
- worst
- plot
- movie
- boring
- supposed
- script
- harry
- reason
- stupid
- waste
- unfortunately
- mess
- ridiculous
- looks
