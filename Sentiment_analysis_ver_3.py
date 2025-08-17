import nltk
import numpy as np
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('movie_reviews')

# Load the data set and store every review as a tuple
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Converts the separate lists of words into a single string (like a sentence)
texts = [" ".join(words) for words, label in documents]

# Extracts the sentiment labels
labels = [label for words, label in documents]

# The added ngram range takes in 3 word features as well as one word for more accurate evaluation of phrases and caps vocabulary to 10000
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,3))
X = vectorizer.fit_transform(texts)

# Debug print
print(X.shape)

# Splits data into training and testing data

# Takes in the matrix, array of labels, size of test data (20% of given data) and a random number to pick the training and test data
# Stratify keeps the class balance similar in both splits
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# Building the classifier using the Logistic Regression model
# Model learns a weight for every ngram then uses this to guess the label for the whole phrase
# The solver assigns the weights based on iterations to make it as accurate as possible
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train, y_train)

# Peak inside the model to see the most positive and negative phrases
feature_names = np.array(vectorizer.get_feature_names_out())

# Weights for "pos" vs "neg"
coefs = model.coef_[0]

top_pos_idx = np.argsort(coefs)[-15:][::-1]
top_neg_idx = np.argsort(coefs)[:15]

print("Top positive n-grams:", feature_names[top_pos_idx])
print("Top negative n-grams:", feature_names[top_neg_idx])

# Make predictions on the test set
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test,y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Create a function to test a custom input
def predict_sentiment(text):
    # Transforms the text to same vector
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# Testing custom input
print(predict_sentiment("I loved this movie!"))
print(predict_sentiment("I hated this movie so much"))
