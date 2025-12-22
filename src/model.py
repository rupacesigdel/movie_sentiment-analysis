import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from features import get_vectorizer


def train_model():
    df = pd.read_csv('C:/Users/a2z/OneDrive/Desktop/sentiment-analysis-project/movie_sentiment-analysis/data/raw/imdb.csv')

    # 2. Features & labels
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 3. Vectorization
    vectorizer = get_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # 5. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\n✅ Model and vectorizer saved successfully!")


if __name__ == "__main__":
    train_model()
