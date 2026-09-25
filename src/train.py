"""Hyperparameter search for RandomForest and GradientBoosting on the Wine dataset."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE

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
