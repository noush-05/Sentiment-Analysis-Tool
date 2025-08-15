import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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


# Converts the text into a numerical format for the model to understand
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

# Splits data into training and testing data

# Takes in the matrix, array of labels, size of test data (20% of given data) and a random number to pick the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Training the model using the Multinomial Naive Bayes method
model = MultinomialNB()
model.fit(X_train, y_train)

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