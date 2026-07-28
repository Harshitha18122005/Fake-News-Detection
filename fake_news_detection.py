# ==========================================
# FAKE NEWS DETECTION USING MACHINE LEARNING
# ==========================================

import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# Load Fake and True datasets
# ------------------------------------------

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
print(fake.head())
print(true.head())

# ------------------------------------------
# Add Labels
# ------------------------------------------

fake["label"] = "FAKE"
true["label"] = "REAL"

# ------------------------------------------
# Merge Datasets
# ------------------------------------------

data = pd.concat([fake, true], ignore_index=True)

# ------------------------------------------
# Shuffle Dataset
# ------------------------------------------

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Loaded Successfully")
print(data.head())
print("\nDataset Shape:", data.shape)

# ------------------------------------------
# Remove Missing Values
# ------------------------------------------

data.dropna(inplace=True)

# ------------------------------------------
# Text Cleaning Function
# ------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------------------
# Clean Text
# ------------------------------------------

data["text"] = data["text"].apply(clean_text)

# ------------------------------------------
# Features and Labels
# ------------------------------------------

X = data["text"]
Y = data["label"]

# ------------------------------------------
# TF-IDF Vectorizer
# ------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

# ------------------------------------------
# Split Dataset
# ------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# ------------------------------------------
# Prediction on Test Data
# ------------------------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(Y_test, prediction)

print("\n=================================")
print("MODEL TRAINED SUCCESSFULLY")
print("=================================")

print("Accuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report")
print(classification_report(Y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test, prediction))

# ------------------------------------------
# User Prediction
# ------------------------------------------

print("\n=================================")
print(" FAKE NEWS DETECTION SYSTEM ")
print("=================================")

while True:

    news = input("\nEnter a News Article (or type EXIT to quit):\n")

    if news.upper() == "EXIT":
        print("\nProgram Closed Successfully.")
        break

    news = clean_text(news)

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    print("\nPrediction Result")

    if result[0] == "FAKE":
        print("FAKE NEWS")
    else:
        print("REAL NEWS")
