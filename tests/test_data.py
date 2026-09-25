"""Unit tests for data loading, splitting, and validation."""
import numpy as np
import pytest

from src.data import (
    EXPECTED_N_FEATURES,
    load_data,
    split_data,
    validate_data,
)


@pytest.fixture
def wine_data():
    """Load the Wine dataset once per test that asks for it."""
    return load_data()


def test_load_data_shape(wine_data):
    X, y = wine_data
    assert X.shape == (178, EXPECTED_N_FEATURES)
    assert len(y) == 178


def test_labels_are_valid_classes(wine_data):
    _, y = wine_data
    assert set(y.unique()) == {0, 1, 2}


def test_validate_data_passes_on_clean_data(wine_data):
    X, y = wine_data
    assert validate_data(X, y) is True


def test_validate_data_rejects_nulls(wine_data):
    X, y = wine_data
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null"):
        validate_data(X_bad, y)


def test_validate_data_rejects_wrong_feature_count(wine_data):
    X, y = wine_data
    X_bad = X.drop(columns=X.columns[0])
    with pytest.raises(ValueError, match="features"):
        validate_data(X_bad, y)


def test_validate_data_rejects_unknown_labels(wine_data):
    X, y = wine_data
    y_bad = y.copy()
    y_bad.iloc[0] = 5
    with pytest.raises(ValueError, match="Unexpected class labels"):
        validate_data(X, y_bad)


def test_split_sizes_are_80_20(wine_data):
    X, y = wine_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 142
    assert len(X_test) == 36


def test_split_has_no_overlap(wine_data):
    X, y = wine_data
    X_train, X_test, _, _ = split_data(X, y)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_stratified(wine_data):
    X, y = wine_data
    _, _, _, y_test = split_data(X, y)
    full = y.value_counts(normalize=True).sort_index()
    test = y_test.value_counts(normalize=True).sort_index()
    assert np.allclose(full.values, test.values, atol=0.02)


def test_split_is_reproducible(wine_data):
    X, y = wine_data
    X_train_1, _, _, _ = split_data(X, y)
    X_train_2, _, _, _ = split_data(X, y)
    assert list(X_train_1.index) == list(X_train_2.index)
