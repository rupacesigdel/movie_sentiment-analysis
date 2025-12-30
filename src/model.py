import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import run_feature_engineering


def train_model():
    print("🔹 Starting Model Training")

    X_train_tfidf, X_test_tfidf, y_train, y_test = run_feature_engineering()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\n📊 Model Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model trained and saved successfully")


if __name__ == "__main__":
    train_model()
