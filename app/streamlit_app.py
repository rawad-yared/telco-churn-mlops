import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st
from src.models.predict_model import (
    load_model,
    predict_single,
    predict_batch,
)


st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    layout="wide",
)

st.title("Telco Churn Prediction Dashboard")
st.markdown(
    """
This app uses the trained Telco Churn model to predict churn probability.

You can enter a single customer's details or upload a CSV for batch scoring.
"""
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])


# ---------- Single Prediction ----------

with tab_single:
    st.subheader("Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            index=0,
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=1,
        )
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0,
        )

    with col2:
        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=120,
            value=5,
            step=1,
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=300.0,
            value=80.0,
            step=1.0,
        )
        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
            index=0,
        )

    with col3:
        partner = st.selectbox(
            "Partner",
            ["No", "Yes"],
            index=0,
        )
        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"],
            index=0,
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=0,
        )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=0,
    )

    if st.button("Predict Churn", type="primary"):
        input_data = {
            # Core contractual & billing features
            "Contract": contract,
            "Internet Service": internet_service,
            "Tenure Months": tenure_months,
            "Monthly Charges": monthly_charges,
            "Paperless Billing": paperless,
            "Payment Method": payment_method,
            # Basic demographics (optional but used by model)
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
        }

        result = predict_single(input_data, model=model)
        prob = result["churn_probability"]
        label = result["churn_prediction"]

        st.markdown("### Prediction Result")
        st.metric(
            label="Churn Probability",
            value=f"{prob:.1%}",
            delta=None,
        )
        # Short, clear prediction summary
        summary = (
            f"**Predicted Churn:** {'Yes' if label == 'Yes' else 'No'} "
            f"**{label}**"
        )
        st.markdown(summary)


# ---------- Batch Prediction ----------

with tab_batch:
    st.subheader("Batch Prediction via CSV")
    st.markdown(
        """
Upload a CSV file where each row is a customer.

**Tips:**
- Use column names similar to the original Telco dataset.
    Examples: `Contract`, `Internet Service`, `Monthly Charges`.
- Missing fields will be handled by the preprocessing pipeline where possible.
"""
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("#### Preview of uploaded data")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction", type="primary"):
            preds_df = predict_batch(df, model=model)

            st.markdown("#### Predictions (first 50 rows)")
            st.dataframe(preds_df.head(50))

            # Download with predictions
            csv_data = preds_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_data,
                file_name=(
                    "telco_churn_predictions.csv"
                ),
                mime="text/csv",
            )
