from pathlib import Path
import mlflow
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"texto", "label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    return df.dropna(subset=["texto", "label"])

def train() -> None:
    df = load_data("data/raw/dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.set_experiment("etapa1_baseline")

    with mlflow.start_run():
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("vectorizer", "tfidf")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

    print({"accuracy": acc, "f1_weighted": f1})

if __name__ == "__main__":
    train()