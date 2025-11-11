from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.predict_model import load_model, predict_single, predict_batch


# Create FastAPI app (must be named `app` for uvicorn app.fastapi_app:app)
app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "API for predicting customer churn using the trained Telco Churn model. "
        "Backed by a sklearn pipeline with feature engineering and MLflow-tracked training."
    ),
    version="1.0.0",
)


# Load model once at startup
@app.on_event("startup")
def load_model_on_startup():
    # Use global so handlers can reuse it
    global model
    model = load_model()


class CustomerPayload(BaseModel):
    """
    Payload wrapper for a single customer.
    Example:
    {
      "data": {
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Monthly Charges": 80,
        "Tenure Months": 5
      }
    }
    """
    data: Dict[str, Any]


class BatchCustomerPayload(BaseModel):
    """
    Payload wrapper for batch prediction.
    Example:
    {
      "records": [
        {...},
        {...}
      ]
    }
    """
    records: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Telco Churn Prediction API is running."}


@app.post("/predict")
def predict_customer(payload: CustomerPayload):
    """
    Predict churn for a single customer.
    """
    result = predict_single(payload.data, model=model)
    return result


@app.post("/predict_batch")
def predict_customers_batch(payload: BatchCustomerPayload):
    """
    Predict churn for multiple customers.
    """
    import pandas as pd

    if not payload.records:
        return {"error": "No records provided."}

    df = pd.DataFrame(payload.records)
    df_out = predict_batch(df, model=model)

    return {"predictions": df_out.to_dict(orient="records")}