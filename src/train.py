"""Hyperparameter search for RandomForest and GradientBoosting on the Wine dataset."""
import os

import mlflow
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, load_data, split_data, validate_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "Wine-Cultivar-Classification"

PARAM_GRIDS = {
    "RandomForest": [
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    ],
    "GradientBoosting": [
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
    ],
}

MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

CV_FOLDS = 5
SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "log_loss": "neg_log_loss",
}


def build_model(model_family, params):
    """Create an untrained model of the given family with the given hyperparameters."""
    model_class = MODEL_CLASSES[model_family]
    return model_class(random_state=RANDOM_STATE, **params)


def cross_validate_model(model, X_train, y_train):
    """Run stratified k-fold CV and return mean train/validation metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True
    )

    metrics = {}
    for name in SCORING:
        train_scores = results[f"train_{name}"]
        val_scores = results[f"test_{name}"]
        if name == "log_loss":
            train_scores, val_scores = -train_scores, -val_scores
        metrics[f"train_{name}"] = float(train_scores.mean())
        metrics[f"val_{name}"] = float(val_scores.mean())
    return metrics


def setup_mlflow(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT_NAME):
    """Point MLflow at the SQLite backend and select (or create) the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment_name)


def run_experiment(model_family, config_index, params, X_train, y_train):
    """Cross-validate one configuration and log it as its own MLflow run."""
    run_name = f"{model_family}-config-{config_index}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model_family": model_family,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_params(params)

        model = build_model(model_family, params)
        metrics = cross_validate_model(model, X_train, y_train)
        mlflow.log_metrics(metrics)

    return run.info.run_id, metrics


def main():
    """Run the full hyperparameter search and log every configuration to MLflow."""
    X, y = load_data()
    validate_data(X, y)
    X_train, _, y_train, _ = split_data(X, y)

    setup_mlflow()

    results = []
    for model_family, grid in PARAM_GRIDS.items():
        for config_index, params in enumerate(grid, start=1):
            run_id, metrics = run_experiment(
                model_family, config_index, params, X_train, y_train
            )
            results.append((model_family, config_index, run_id, metrics))
            print(
                f"{model_family}-config-{config_index}: "
                f"val_f1_macro={metrics['val_f1_macro']:.4f}  run_id={run_id}"
            )
    return results


if __name__ == "__main__":
    main()
