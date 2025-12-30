import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def get_vectorizer():
    """
    Create and return a TF-IDF vectorizer
    """
    return TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )


def run_feature_engineering(
    data_path="C:/Users/a2z/OneDrive/Desktop/sentiment-analysis-project/movie_sentiment-analysis/data/processed/cleaned_data.csv",
    model_dir="models"
):
    """
    Perform feature engineering: load data, split, vectorize, and save vectorizer
    """

    print("🔹 Starting Feature Engineering")

    df = pd.read_csv(data_path)

    df["sentiment"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = get_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Feature engineering completed")
    print("✅ TF-IDF vectorizer saved")

    return X_train_tfidf, X_test_tfidf, y_train, y_test


if __name__ == "__main__":
    run_feature_engineering()
