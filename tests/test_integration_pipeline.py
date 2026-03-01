from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_training_pipeline_integration_from_csv(nb, sample_csv_path: Path) -> None:
    config = nb.ModelConfig()
    df = nb.load_dataset(sample_csv_path)
    x_train, x_test, y_train, y_test = nb.split_dataset(df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)
    y_pred = model.predict(x_test_vec)
    metrics = nb.evaluate_model(y_test, y_pred)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_inference_pipeline_integration(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)

    new_inputs = pd.Series([
        "official statement confirms policy",
        "viral hoax about miracle treatment",
    ])
    new_features = vectorizer.transform(new_inputs)
    predictions = model.predict(new_features)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)


def test_end_to_end_with_repository_dataset_subset(nb) -> None:
    repo_csv = Path(__file__).resolve().parents[1] / "news" / "news.csv"
    df = nb.load_dataset(repo_csv).head(200)

    config = nb.ModelConfig()
    x_train, x_test, y_train, y_test = nb.split_dataset(df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)
    y_pred = model.predict(x_test_vec)
    metrics = nb.evaluate_model(y_test, y_pred)

    assert len(y_pred) == len(y_test)
    assert metrics["confusion_matrix"].shape == (2, 2), "Binary classification must produce a 2x2 confusion matrix"
    assert 0.0 <= metrics["accuracy"] <= 1.0
