"""MLOps quality gate: the champion model must be accurate, fast, and well-formed."""
import time

import mlflow
import numpy as np
import pytest
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient

from src import train
from src.data import EXPECTED_N_FEATURES, load_data, split_data
from src.evaluate import load_champion

F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0
VALID_CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    """Train and register a champion in a temporary MLflow store, then load it."""
    tmp_dir = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{tmp_dir / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(
        train.EXPERIMENT_NAME, artifact_location=(tmp_dir / "artifacts").as_uri()
    )
    train.main(tracking_uri=tracking_uri)

    client = MlflowClient()
    version = client.get_model_version_by_alias(
        train.REGISTERED_MODEL_NAME, train.CHAMPION_ALIAS
    )
    val_f1 = client.get_run(version.run_id).data.metrics["val_f1_macro"]

    yield {
        "model": load_champion(),
        "val_f1_macro": val_f1,
        "run_id": version.run_id,
    }

    mlflow.set_tracking_uri(train.TRACKING_URI)


@pytest.fixture(scope="module")
def test_split():
    """The held-out test split, identical to the one used in evaluation."""
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    return X_test, y_test


def test_metric_gate_validation_f1(champion):
    val_f1 = champion["val_f1_macro"]
    assert val_f1 >= F1_THRESHOLD, (
        f"Validation macro F1 {val_f1:.4f} is below the {F1_THRESHOLD} gate."
    )


def test_latency_gate_batch_inference(champion, test_split):
    model = champion["model"]
    X_test, _ = test_split

    model.predict(X_test)
    timings_ms = []
    for _ in range(5):
        start = time.perf_counter()
        model.predict(X_test)
        timings_ms.append((time.perf_counter() - start) * 1000)
    median_ms = float(np.median(timings_ms))

    assert median_ms <= LATENCY_THRESHOLD_MS, (
        f"Batch inference took {median_ms:.2f} ms, above the {LATENCY_THRESHOLD_MS} ms gate."
    )


def test_output_schema_gate_class_indices(champion, test_split):
    model = champion["model"]
    X_test, _ = test_split
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(np.unique(predictions)).issubset(VALID_CLASSES), (
        f"Unexpected predicted classes: {set(np.unique(predictions)) - VALID_CLASSES}"
    )


def test_signature_expects_all_features(champion):
    model_info = get_model_info(f"runs:/{champion['run_id']}/model")
    signature = model_info.signature
    assert signature is not None, "Registered champion has no signature."
    assert len(signature.inputs.input_names()) == EXPECTED_N_FEATURES
