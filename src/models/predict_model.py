from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def _get_latest_model_path() -> Path:
    """
    Return the most recent saved model in models/.
    """
    model_files = sorted(
        MODELS_DIR.glob("best_model_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not model_files:
        raise FileNotFoundError(
            f"No saved models found in {MODELS_DIR}. "
            f"Run `python -m src.models.train_model` first."
        )
    return model_files[0]


def load_model(model_path: Path = None) -> Pipeline:
    """
    Load the latest saved sklearn Pipeline (preprocessing + model).
    """
    if model_path is None:
        model_path = _get_latest_model_path()

    print(f"Loading model pipeline from: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError("Loaded object is not a sklearn Pipeline.")
    return model


# ---------- Schema alignment helpers ----------


def _get_expected_input_columns(model: Pipeline):
    """
    Infer which input columns the pipeline expects BEFORE preprocessing.

    We inspect the internal preprocessor pipeline created in training:
    preprocessor = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer),
        ("preprocess", ColumnTransformer(...))
    ])

    The ColumnTransformer stores the column names it was fitted on.
    We treat those as the required columns and ensure they exist at inference.
    """
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "named_steps"):
        # Fallback: don't attempt schema alignment
        return None

    if "preprocess" not in preprocessor.named_steps:
        return None

    ct = preprocessor.named_steps["preprocess"]

    expected_cols = set()
    for _, _, cols in ct.transformers_:
        if cols is None or cols == "drop":
            continue
        # 'passthrough' is not used here; if it were, we'd handle separately.
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            expected_cols.update(list(cols))

    if not expected_cols:
        return None

    return sorted(expected_cols)


def _prepare_input_df(
    data: Union[Dict[str, Union[str, int, float]], pd.DataFrame],
    model: Pipeline,
) -> pd.DataFrame:
    """
    Take raw input (dict or DataFrame) and:
        - Convert to DataFrame if needed.
        - Ensure all columns expected by the trained pipeline exist.
            Missing columns are filled with NaN so that imputers and logic
            can handle them.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a dict or pandas DataFrame.")

    expected_cols = _get_expected_input_columns(model)

    if expected_cols is not None:
        for col in expected_cols:
            if col not in df.columns:
                # Missing features -> set to NaN (neutral / imputed later)
                df[col] = np.nan

        # Ensure we don't accidentally drop any provided columns;
        # ColumnTransformer selects by name, so extra columns are harmless.
        # No need to reorder strictly.
    return df


# ---------- Public prediction functions ----------


def predict_single(
    input_dict: Dict[str, Union[str, int, float]],
    model: Pipeline = None,
) -> Dict[str, Union[str, float]]:
    """
    Predict churn for a single customer.
    Returns:
      - churn_prediction: 'Yes' or 'No'
      - churn_probability: probability of churn (float)
    """
    if model is None:
        model = load_model()

    df = _prepare_input_df(input_dict, model)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    prob = float(model.predict_proba(df)[0][1])
    label = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_prediction": label,
        "churn_probability": round(prob, 4),
    }


def predict_batch(
    df: pd.DataFrame,
    model: Pipeline = None,
) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Input:
      - df: DataFrame with one row per customer.

    Output:
      - original columns +
      - churn_probability
      - churn_prediction
    """
    if model is None:
        model = load_model()

    df_prepared = _prepare_input_df(df, model)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    probs = model.predict_proba(df_prepared)[:, 1]
    labels = np.where(probs >= 0.5, "Yes", "No")

    out = df.copy()
    out["churn_probability"] = np.round(probs, 4)
    out["churn_prediction"] = labels
    return out
