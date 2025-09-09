# Sentiment Analysis Tool
A supervised machine learning model for sentiment analysis, trained on the NLTK movie reviews dataset. The model predicts whether a text review is positive or negative.

## Context
This project represents my first hands-on attempt at training a machine learning model using a well-established dataset. I chose sentiment analysis because it has broad applications — from analyzing product reviews to detecting harmful content online. In particular, sentiment classification can contribute to efforts in identifying bullying or hate speech by analyzing linguistic patterns, helping to reduce harm in digital spaces.

To deepen my understanding, I implemented three versions of the tool, each with a different model. This iterative approach allowed me to compare model behavior, experiment with feature extraction, and evaluate the trade-offs between accuracy, precision, and recall.

## Features
- Text preprocessing with TF-IDF: Converts raw text into numerical feature vectors for model training.
- N-gram support: Captures single words as well as multi-word phrases to improve context understanding.
- Custom prediction function: Allows users to input their own text and receive a sentiment prediction.
- Iterative model versions:
  - Version 1 (Baseline): Naive Bayes with unigrams
  - Version 2: Naive Bayes with unigrams + bigrams
  - Version 3: Logistic Regression with unigrams, bigrams, and trigrams

## Results 
| Version | Model           | Accuracy | Neg Recall | Pos Recall | Comments   |
|---------|-----------------|----------|------------|------------|------------|
| 01      | NB (unigrams)   | 0.7925   | 0.85       | 0.74       | too pessimistic
| 02      | NB (unigrams+bigrams)     | 0.8      | 0.84       | 0.76       | Slight improvement in positive recall
| 03      | Logistic Regression | 0.8225 | 0.80     | 0.84       | Significant improvement in positive recall

## Top n-grams in version 3
Top positive n-grams:
great, life, war, family, excellent, truman, best, perfect, mulan, world, overall, perfectly, jackie, performance, terrific

Top negative n-grams:
bad, worst, plot, movie, boring, supposed, script, harry, reason, stupid, waste, unfortunately, mess, ridiculous, looks

## Quick Start Guide
Use the following steps to run the project:

1. Clone the repository - git clone https://github.com/noush-05/Sentiment-Analysis-Tool.git
cd Sentiment-Analysis-Tool

2. Install dependencies - pip install -r requirements.txt

3.  Download the NLTK movie_reviews dataset -
   import nltk
   nltk.download('movie_reviews')

4. Run a model version -
  e.g. python src/version1.py

5. Test with your own statements -
   from src.version3 import predict_sentiment
   print(predict_sentiment("I loved this movie!"))   
   print(predict_sentiment("The plot was boring."))  

