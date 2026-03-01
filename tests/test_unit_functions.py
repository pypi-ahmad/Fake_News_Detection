from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_model_config_defaults(nb) -> None:
    config = nb.ModelConfig()
    assert config.test_size == 0.2
    assert config.random_state == 7
    assert config.max_df == 0.7
    assert config.max_iter == 100
    assert config.stop_words == "english"


def test_load_dataset_success(nb, sample_csv_path: Path, sample_df: pd.DataFrame) -> None:
    loaded = nb.load_dataset(sample_csv_path)
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_dataset_missing_file_raises(nb, tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        nb.load_dataset(missing_path)


def test_load_dataset_missing_required_columns_raises(nb, tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["x"], "label": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        nb.load_dataset(bad_csv)


def test_load_dataset_binary_file_raises_value_error(nb, tmp_path: Path) -> None:
    binary_path = tmp_path / "bad.jpg"
    binary_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    with pytest.raises(ValueError):
        nb.load_dataset(binary_path)


def test_split_dataset_returns_expected_partitions(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, y_test = nb.split_dataset(sample_df, config)
    assert len(x_train) + len(x_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)
    assert len(x_test) == 2
    assert len(y_test) == 2
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_build_vectorizer_uses_config(nb) -> None:
    config = nb.ModelConfig(max_df=0.55, stop_words="english")
    vectorizer = nb.build_vectorizer(config)
    params = vectorizer.get_params()
    assert params["max_df"] == 0.55
    assert params["stop_words"] == "english"


def test_vectorize_text_returns_matrices(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, _, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    assert x_train_vec.shape[0] == len(x_train)
    assert x_test_vec.shape[0] == len(x_test)


def test_vectorize_text_empty_input_raises(nb) -> None:
    config = nb.ModelConfig()
    vectorizer = nb.build_vectorizer(config)
    empty_series = pd.Series([], dtype="object")
    with pytest.raises(ValueError):
        nb.vectorize_text(vectorizer, empty_series, empty_series)


def test_vectorize_text_handles_null_text(nb) -> None:
    config = nb.ModelConfig()
    vectorizer = nb.build_vectorizer(config)
    x_train = pd.Series(["valid", None, "more", np.nan])
    x_test = pd.Series([None, "news"]) 
    x_train_vec, x_test_vec = nb.vectorize_text(vectorizer, x_train, x_test)
    assert x_train_vec.shape[0] == 4
    assert x_test_vec.shape[0] == 2


def test_train_model_returns_classifier(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, _ = nb.vectorize_text(vectorizer, x_train, x_test)
    model = nb.train_model(x_train_vec, y_train, config)
    assert model.__class__.__name__ == "PassiveAggressiveClassifier"


def test_train_model_invalid_shape_raises(nb, sample_df: pd.DataFrame) -> None:
    config = nb.ModelConfig()
    x_train, x_test, y_train, _ = nb.split_dataset(sample_df, config)
    vectorizer = nb.build_vectorizer(config)
    x_train_vec, _ = nb.vectorize_text(vectorizer, x_train, x_test)
    wrong_labels = y_train.iloc[:-1]
    with pytest.raises(ValueError):
        nb.train_model(x_train_vec, wrong_labels, config)


def test_evaluate_model_returns_metrics_dict(nb) -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
    metrics = nb.evaluate_model(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "confusion_matrix"}
    assert isinstance(metrics["accuracy"], float)
    assert metrics["confusion_matrix"].shape == (2, 2)
