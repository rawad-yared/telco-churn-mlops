from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---- Paths & core columns ---- #

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "telco_churn_clean.parquet"
)

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn Label"  # ensured during ingestion; adjust if needed

# Columns that are clearly leakage / post-outcome or business-derived
LEAKAGE_COLUMNS = [
    "Churn Reason",
    "Churn Score",
    "Churn Category",
    "Churn Value",
    "CLTV",
]


# ---- Custom Feature Engineering Transformer ---- #

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create domain-specific features in a sklearn-compatible way.
    This runs inside the pipeline so it's consistently applied
    in both training and inference.
    """

    def __init__(self):
        # Candidate service columns to count for Total_Services
        self.service_cols_candidates = [
            "Phone Service",
            "PhoneService",
            "Multiple Lines",
            "MultipleLines",
            "Internet Service",
            "InternetService",
            "Online Security",
            "OnlineSecurity",
            "Online Backup",
            "OnlineBackup",
            "Device Protection",
            "DeviceProtection",
            "Tech Support",
            "TechSupport",
            "Streaming TV",
            "StreamingTV",
            "Streaming Movies",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        # Remember which of these actually exist
        self.service_cols_ = [
            c for c in self.service_cols_candidates if c in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

    # Ensure all expected service columns exist; if missing,
    # default to "No" so they don't artificially increase
    # Total_Services.
        for col in self.service_cols_:
            if col not in X.columns:
                X[col] = "No"

    # --- Total_Services: count "Yes" across available service
    # columns --- #
        if self.service_cols_:
            def count_services(row):
                count = 0
                for v in row:
                    vs = str(v).strip().lower()
                    if vs == "yes":
                        count += 1
                return count

            X["Total_Services"] = (
                X[self.service_cols_]
                .apply(count_services, axis=1)
                .astype("float64")
            )

        # --- Avg_Charge_Per_Service: MonthlyCharges / Total_Services --- #
        if "Monthly Charges" in X.columns and "Total_Services" in X.columns:
            denom = X["Total_Services"].replace(0, np.nan)
            X["Avg_Charge_Per_Service"] = X["Monthly Charges"] / denom
            X["Avg_Charge_Per_Service"] = (
                X["Avg_Charge_Per_Service"].fillna(0.0)
            )

        # --- TenureGroup: bins of Tenure --- #
        tenure_col = None
        if "Tenure Months" in X.columns:
            tenure_col = "Tenure Months"
        elif "tenure" in X.columns:
            tenure_col = "tenure"

        if tenure_col is not None:
            X["TenureGroup"] = pd.cut(
                X[tenure_col],
                bins=[-1, 6, 12, 24, 48, np.inf],
                labels=["New", "0-1y", "1-2y", "2-4y", "4y+"],
            )

        # --- ARPU: TotalCharges / Tenure --- #
        total_charges_col = None
        if "Total Charges" in X.columns:
            total_charges_col = "Total Charges"
        elif "TotalCharges" in X.columns:
            total_charges_col = "TotalCharges"

        if tenure_col is not None and total_charges_col is not None:
            tenure_safe = X[tenure_col].replace(0, np.nan)
            X["ARPU"] = X[total_charges_col] / tenure_safe
            X["ARPU"] = X["ARPU"].fillna(0.0)

        # --- Contract_InternetInteraction: interaction feature --- #
        contract_col = "Contract" if "Contract" in X.columns else None
        internet_col = (
            "Internet Service"
            if "Internet Service" in X.columns
            else "InternetService"
            if "InternetService" in X.columns
            else None
        )

        if contract_col and internet_col:
            X["Contract_InternetInteraction"] = (
                X[contract_col].astype(str) + "_" + X[internet_col].astype(str)
            )

        return X


# ---- Helpers to clean feature space ---- #

def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ID, target, and known leakage columns from the feature space.
    """
    cols_to_drop = [ID_COLUMN, TARGET_COLUMN] + [
        c for c in LEAKAGE_COLUMNS if c in df.columns
    ]
    return df.drop(
        columns=[c for c in cols_to_drop if c in df.columns],
        errors="ignore",
    )


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical features after feature engineering.
    """
    numeric_features = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = [c for c in df.columns if c not in numeric_features]
    return numeric_features, categorical_features


# ---- Main builder used by training & inference ---- #

def build_preprocessing_pipeline(df_sample: pd.DataFrame) -> Pipeline:
    """
    Build a full sklearn Pipeline given a sample dataframe that includes
    the columns from processed data.

    Steps:
      1. Drop ID, target, and leakage columns.
      2. Apply FeatureEngineer.
      3. Apply:
         - median impute + standard scale for numeric,
         - most_frequent impute + one-hot for categorical.
    """
    # 1) Remove non-features
    X = drop_non_features(df_sample)

    # 2) Run feature engineering once on sample to detect resulting columns
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)

    # 3) Infer types on engineered space
    numeric_features, categorical_features = infer_feature_types(X_fe)

    # 4) Define transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 5) Full pipeline: FeatureEngineer -> ColumnTransformer
    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("preprocess", preprocessor),
        ]
    )

    return full_pipeline


# ---- Convenience functions for training script ---- #

def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load the processed parquet created by src.data.load_data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            f"Run `python -m src.data.load_data` first."
        )
    return pd.read_parquet(path)


def get_feature_matrix_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given the processed dataframe, return (X, y) before sklearn transforms.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataframe."
        )

    y = df[TARGET_COLUMN].copy()
    X = drop_non_features(df)

    return X, y
