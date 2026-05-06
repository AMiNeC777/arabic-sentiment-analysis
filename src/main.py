from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.metrics import classification_metrics
from features.tfidf import TFIDFVectorizer
from models.ml.logistic_regression import LogisticRegressionModel
from preprocessing.clean_text import clean_text
from preprocessing.normalize import normalize_arabic
from utils.config import load_config
from utils.logger import get_logger


def preprocess_series(text_series: pd.Series) -> pd.Series:
    return text_series.fillna("").map(clean_text).map(normalize_arabic)


def run_baseline_pipeline(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    logger = get_logger(log_dir=config["paths"]["logs_dir"])

    processed_dir = Path(config["paths"]["processed_data_dir"])
    dataset_path = processed_dir / "dataset.csv"

    if not dataset_path.exists():
        logger.warning(
            "No dataset found at %s. Add `data/processed/dataset.csv` with text/label columns.",
            dataset_path,
        )
        return

    df = pd.read_csv(dataset_path)
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    X = preprocess_series(df[text_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )

    vectorizer = TFIDFVectorizer(
        max_features=config["features"]["max_features"],
        ngram_range=tuple(config["features"]["ngram_range"]),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegressionModel().fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = classification_metrics(y_test, y_pred, average=config["evaluation"]["average"])
    logger.info("Baseline metrics: %s", metrics)


if __name__ == "__main__":
    run_baseline_pipeline()
