import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    load_processed_data,
    get_feature_matrix_and_target,
    build_preprocessing_pipeline,
)

warnings.filterwarnings("ignore")

# ---- Paths & constants ---- #

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "telco-churn-mlops"


# ---- Helpers ---- #

def encode_target(y_raw) -> np.ndarray:
    """
    Encode target labels into binary 0/1.
    Assumes churn positive class is some variant of 'Yes'.
    """
    return (y_raw.astype(str).str.strip().str.lower() == "yes").astype(int).values


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Compute core evaluation metrics.
    """
    metrics = {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def train_and_evaluate(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Build full pipeline (preprocessing + model), train, evaluate and return both.
    """
    # Build preprocessing pipeline on training data only
    preprocessor = build_preprocessing_pipeline(X_train)

    # Full pipeline: preprocessing + classifier
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", model),
        ]
    )

    # Fit
    clf_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = clf_pipeline.predict(X_test)

    # Probabilities if available
    if hasattr(clf_pipeline.named_steps["clf"], "predict_proba"):
        y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Confusion matrix (as individual values for logging)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update(
        {
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    )

    return clf_pipeline, metrics


def log_run_to_mlflow(
    model_name: str,
    model,
    pipeline: Pipeline,
    metrics: Dict[str, float],
):
    """
    Log parameters, metrics, and model artifact to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Model hyperparameters (basic introspection)
        if hasattr(model, "get_params"):
            params = model.get_params()
            # Log only simple params to avoid clutter
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

        # Log metrics
        for m_name, m_value in metrics.items():
            mlflow.log_metric(m_name, float(m_value))

        # Log model pipeline
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=None,  # registry optional for now
        )


# ---- Main training routine ---- #

def main():
    # Configure MLflow experiment (local ./mlruns by default)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load processed data
    df = load_processed_data()

    # 2. Split into features & target
    X, y_raw = get_feature_matrix_and_target(df)
    y = encode_target(y_raw)

    # 3. Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Define candidate models
    models = {
        "log_reg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    best_model_name = None
    best_pipeline = None
    best_metrics = None
    best_f1 = -1.0

    # 5. Train and log each candidate
    for name, model in models.items():
        print(f"\n=== Training model: {name} ===")

        pipeline, metrics = train_and_evaluate(
            model_name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        print(f"Metrics for {name}: {metrics}")

        # Log to MLflow
        log_run_to_mlflow(name, model, pipeline, metrics)

        # Track best by F1
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    # 6. Persist best model pipeline to disk
    if best_pipeline is not None:
        best_model_path = MODELS_DIR / f"best_model_{best_model_name}.joblib"
        joblib.dump(best_pipeline, best_model_path)
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1: {best_f1:.4f}")
        print(f"Saved best model pipeline to: {best_model_path}")
    else:
        raise RuntimeError("No model was successfully trained.")


if __name__ == "__main__":
    main()