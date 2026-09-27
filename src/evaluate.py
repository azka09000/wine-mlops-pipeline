"""Load the champion model from the MLflow registry and evaluate it on the test split."""
import time

import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_data, split_data, validate_data
from src.train import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, setup_mlflow

CHAMPION_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


def load_champion(model_uri=CHAMPION_URI):
    """Load the model version that the champion alias currently points to."""
    return mlflow.sklearn.load_model(model_uri)


def evaluate_model(model, X_test, y_test):
    """Compute test metrics and the time taken to predict the whole batch."""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    batch_inference_ms = (time.perf_counter() - start) * 1000

    y_proba = model.predict_proba(X_test)
    return {
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_log_loss": float(log_loss(y_test, y_proba, labels=[0, 1, 2])),
        "batch_inference_ms": batch_inference_ms,
    }


def main():
    """Evaluate the registered champion on the held-out test split."""
    setup_mlflow()

    X, y = load_data()
    validate_data(X, y)
    _, X_test, _, y_test = split_data(X, y)

    model = load_champion()
    metrics = evaluate_model(model, X_test, y_test)

    print(f"Evaluated {CHAMPION_URI} on {len(X_test)} test samples:")
    for name, value in metrics.items():
        print(f"  {name:20s} {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
