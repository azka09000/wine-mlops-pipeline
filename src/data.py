"""Data loading, splitting, and validation for the Wine cultivar dataset."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_N_FEATURES = 13
EXPECTED_CLASSES = {0, 1, 2}


def load_data():
    """Load the Wine dataset as a pandas DataFrame (X) and Series (y)."""
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split so each class keeps the same proportion."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def validate_data(X, y):
    """Raise ValueError if the data is not in the expected shape or state."""
    if X.isnull().values.any():
        raise ValueError("Features contain null values.")
    if y.isnull().values.any():
        raise ValueError("Target contains null values.")
    if X.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features, got {X.shape[1]}."
        )
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
    unexpected = set(y.unique()) - EXPECTED_CLASSES
    if unexpected:
        raise ValueError(f"Unexpected class labels: {unexpected}.")
    return True
