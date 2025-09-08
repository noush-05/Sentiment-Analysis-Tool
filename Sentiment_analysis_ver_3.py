import nltk
import numpy as np
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download the dataset
nltk.download('movie_reviews')

# Load the data set and store every review as a tuple
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert the separate lists of words into a single string (like a sentence)
texts = [" ".join(words) for words, label in documents]

# Extract sentiment labels
labels = [label for words, label in documents]

# Vectorize text with TF-IDF (ngrams up to 3 words, max 10,000 features)
vectorizer = TfidfVectorizer(
    # Ignores english filler words
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 3)
)
X = vectorizer.fit_transform(texts)

# Split data into training and testing (80/20 split, stratified for balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# Build the classifier by iterating the process of guessing weights and taking an average
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Look at most positive and negative n-grams
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

# Takes the 15 most positive and negative n gram signals to output
top_pos_idx = np.argsort(coefs)[-15:][::-1]
top_neg_idx = np.argsort(coefs)[:15]

print("Top positive n-grams:", feature_names[top_pos_idx])
print("Top negative n-grams:", feature_names[top_neg_idx])

# Make predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Custom prediction function
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction


# Test custom inputs
print(predict_sentiment("I loved this movie!"))
print(predict_sentiment("I hated this movie so much"))
