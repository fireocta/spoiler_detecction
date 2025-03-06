import json
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('punkt')

# Load the datasets
def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)

with open("/home/codespace/.cache/kagglehub/datasets/rmisra/imdb-spoiler-dataset/versions/1/IMDB_movie_details.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)

reviews = list(parse_data("/home/codespace/.cache/kagglehub/datasets/rmisra/imdb-spoiler-dataset/versions/1/IMDB_reviews.json"))

# Convert JSON to DataFrame
meta_df = pd.DataFrame(metadata)
review_df = pd.DataFrame(reviews)

# Merge reviews with metadata on movie_id
data = review_df.merge(meta_df, on="movie_id", how="left")

# Use only relevant columns
data = data[['review_text', 'plot_summary', 'plot_synopsis', 'is_spoiler']]

# Combine review_text with plot_summary and plot_synopsis for better context
data['text'] = data['review_text'] + " " + data['plot_summary'] + " " + data['plot_synopsis']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Apply text preprocessing
data["processed_text"] = data["text"].apply(preprocess_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["processed_text"])
y = data["is_spoiler"]  # 1 = spoiler, 0 = non-spoiler

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
svm_model = SVC()
nb_model = MultinomialNB()

svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

# Evaluation
y_pred_svm = svm_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)

print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Naïve Bayes Model Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nNaïve Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# Save models
import joblib
joblib.dump(svm_model, "svm_spoiler_model.pkl")
joblib.dump(nb_model, "nb_spoiler_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
