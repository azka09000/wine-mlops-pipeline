"""Hyperparameter search for RandomForest and GradientBoosting on the Wine dataset."""
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, load_data, split_data, validate_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"

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
    """Cross-validate one configuration, fit it on the full train split, and log it."""
    run_name = f"{model_family}-config-{config_index}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model_family": model_family,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_params(params)

        metrics = cross_validate_model(
            build_model(model_family, params), X_train, y_train
        )
        mlflow.log_metrics(metrics)

        model = build_model(model_family, params)
        model.fit(X_train, y_train)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return run.info.run_id, metrics


def select_best_run(experiment_id):
    """Return the logged run with the highest val macro F1 (tie-break: lowest val log loss)."""
    client = MlflowClient()
    best_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_f1_macro DESC", "metrics.val_log_loss ASC"],
        max_results=1,
    )
    if not best_runs:
        raise RuntimeError("No finished runs found to select a champion from.")
    return best_runs[0]


def register_champion(run_id):
    """Register the run's model as WineClassifier and point the champion alias at it."""
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=REGISTERED_MODEL_NAME
    )
    client = MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_version.version
    )
    return model_version


def main():
    """Run the search, log every configuration, and register the champion model."""
    X, y = load_data()
    validate_data(X, y)
    X_train, _, y_train, _ = split_data(X, y)

    experiment = setup_mlflow()

    for model_family, grid in PARAM_GRIDS.items():
        for config_index, params in enumerate(grid, start=1):
            run_id, metrics = run_experiment(
                model_family, config_index, params, X_train, y_train
            )
            print(
                f"{model_family}-config-{config_index}: "
                f"val_f1_macro={metrics['val_f1_macro']:.4f}  run_id={run_id}"
            )

    best_run = select_best_run(experiment.experiment_id)
    model_version = register_champion(best_run.info.run_id)
    print(
        f"\nChampion: {best_run.info.run_name} "
        f"(val_f1_macro={best_run.data.metrics['val_f1_macro']:.4f}, "
        f"val_log_loss={best_run.data.metrics['val_log_loss']:.4f})"
    )
    print(
        f"Registered as {REGISTERED_MODEL_NAME} version {model_version.version} "
        f"with alias '{CHAMPION_ALIAS}'"
    )


if __name__ == "__main__":
    main()
