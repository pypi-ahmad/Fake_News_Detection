from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_model_artifacts_roundtrip_loads_correctly(nb, sample_df: pd.DataFrame, tmp_path: Path) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)

    artifact_path = tmp_path / "model.pkl"
    nb.save_model_artifacts(model, vectorizer, artifact_path)
    loaded_model, loaded_vectorizer = nb.load_model_artifacts(artifact_path)

    loaded_predictions = loaded_model.predict(loaded_vectorizer.transform(x_test))
    original_predictions = model.predict(x_test_vec)

    assert loaded_model.__class__.__name__ == "PassiveAggressiveClassifier"
    assert loaded_vectorizer.__class__.__name__ == "TfidfVectorizer"
    assert np.array_equal(loaded_predictions, original_predictions)


def test_prediction_output_shape_and_type(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)
    predictions = model.predict(x_test_vec)

    assert isinstance(predictions, np.ndarray)
    assert predictions.ndim == 1
    assert predictions.shape[0] == x_test_vec.shape[0]


def test_split_dataset_invalid_dataframe_raises(nb) -> None:
    bad_df = pd.DataFrame({"title": ["missing text and label"]})
    config = nb.ModelConfig()
    with pytest.raises(ValueError):
        nb.split_dataset(bad_df, config)


def test_split_dataset_null_labels_raises(nb) -> None:
    df = pd.DataFrame({
        "text": ["article one", "article two", "article three", "article four"],
        "label": [0, None, 1, 0],
    })
    config = nb.ModelConfig()
    with pytest.raises(ValueError, match="null values"):
        nb.split_dataset(df, config)


def test_evaluate_model_mismatched_lengths_raises(nb) -> None:
    y_true = pd.Series([0, 1, 0])
    y_pred = np.array([0, 1], dtype=object)
    with pytest.raises(ValueError):
        nb.evaluate_model(y_true, y_pred)


def test_missing_model_file_raises_on_load(nb, tmp_path: Path) -> None:
    missing = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError):
        nb.load_model_artifacts(missing)


def test_corrupted_model_file_raises_on_load(nb, tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.pkl"
    corrupted.write_bytes(b"not a valid pickle payload")
    with pytest.raises(ValueError):
        nb.load_model_artifacts(corrupted)
