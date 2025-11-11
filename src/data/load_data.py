import pandas as pd
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.parquet"

# Canonical names we will enforce
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn Label"

# Possible variants in raw data
ID_CANDIDATES = ["customerID", "Customer ID", "CustomerID"]
TARGET_CANDIDATES = ["Churn Label", "Churn", "Churn_Label"]

# Columns we will try to coerce to numeric if present
NUMERIC_CANDIDATES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Churn Value",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset from an Excel file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_excel(path)

    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Standardize key columns (ID + Target) to canonical names
    df = _standardize_key_columns(df)

    return df


def _standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename different variants of ID and target column names to canonical ones.
    """
    cols = list(df.columns)

    # Standardize ID column
    id_found = None
    for cand in ID_CANDIDATES:
        if cand in cols:
            id_found = cand
            break
    if id_found:
        df = df.rename(columns={id_found: ID_COLUMN})

    # Standardize target column
    target_found = None
    for cand in TARGET_CANDIDATES:
        if cand in cols:
            target_found = cand
            break
    if target_found and target_found != TARGET_COLUMN:
        df = df.rename(columns={target_found: TARGET_COLUMN})

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light, safe cleaning:
    - Ensure ID and target columns exist.
    - Coerce selected numeric-like columns.
    - Drop rows without target label.
    """
    missing = []
    if ID_COLUMN not in df.columns:
        missing.append(ID_COLUMN)
    if TARGET_COLUMN not in df.columns:
        missing.append(TARGET_COLUMN)

    if missing:
        raise ValueError(
            f"Missing required columns after standardization: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce numeric candidates where present
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target label
    df = df.dropna(subset=[TARGET_COLUMN])

    # Reset index after drops
    df = df.reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """
    Save cleaned dataframe to Parquet for efficient downstream use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_ingestion() -> None:
    """
    Orchestrate the ingestion:
    - Load raw Excel
    - Standardize / basic clean
    - Save to processed/
    """
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df_raw = load_raw_data()
    print(f"Raw shape: {df_raw.shape}")

    df_clean = basic_cleaning(df_raw)
    print(f"Cleaned shape: {df_clean.shape}")

    print(f"Saving processed data to: {PROCESSED_DATA_PATH}")
    save_processed_data(df_clean)
    print("Ingestion step completed successfully.")


if __name__ == "__main__":
    run_ingestion()