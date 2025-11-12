# 📡 Telco Churn MLOps Project

[![CI](https://github.com/rawad-yared/telco-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/rawad-yared/telco-churn-mlops/actions/workflows/ci.yml)

End-to-end churn prediction project using the **IBM Telco Customer Churn** dataset, built to demonstrate practical **MLOps** concepts:

- Reproducible data & feature pipelines  
- Model training & evaluation with scikit-learn  
- Experiment tracking with MLflow  
- Containerized deployment with Docker  
- FastAPI API for online inference  
- Streamlit dashboard for interactive predictions  

---

## 🧭 1. Project Overview

The goal is to predict whether a telecom customer will **churn** (leave) based on their contract, services, and billing behavior.

**Tech stack:**
- Python 3.12  
- pandas, numpy, scikit-learn  
- MLflow for experiment tracking  
- FastAPI + Uvicorn for model serving  
- Streamlit for the dashboard  
- Docker for containerization  

---

## 🗂️ 2. Repository Structure

telco-churn-mlops/
├─ app/
│  ├─ init.py
│  ├─ fastapi_app.py        # FastAPI app exposing /health, /predict, /predict_batch
│  └─ streamlit_app.py      # Streamlit dashboard (single + batch predictions)
├─ src/
│  ├─ data/
│  │  └─ load_data.py       # Ingestion from XLSX → cleaned parquet
│  ├─ features/
│  │  └─ build_features.py  # Feature engineering + preprocessing pipeline
│  └─ models/
│     ├─ train_model.py     # Train + evaluate + log with MLflow + save best model
│     └─ predict_model.py   # Load best model + single/batch inference helpers
├─ data/
│  ├─ raw/                  # (gitignored) place Telco_customer_churn.xlsx here
│  └─ processed/            # (gitignored) parquet saved after cleaning
├─ models/                  # (gitignored) trained model(s)
├─ mlruns/                  # (gitignored) MLflow experiment logs
├─ Dockerfile               # Container definition for FastAPI API
├─ requirements.txt
├─ .gitignore
└─ README.md

**Note:**  
`data/`, `models/`, and `mlruns/` are **ignored by Git** — they’re generated locally by the pipeline.

---

## ⚙️ 3. Setup Instructions

### 🪄 Clone the repository

```bash
git clone https://github.com/<your-username>/telco-churn-mlops.git
cd telco-churn-mlops

🧱 Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

📦 Install dependencies

pip install --upgrade pip
pip install -r requirements.txt


⸻

📊 4. Data Setup

Download the Telco Customer Churn dataset (Telco_customer_churn.xlsx)
and place it under:

data/raw/Telco_customer_churn.xlsx


⸻

🧩 5. Run the Pipeline (Local)

Step 1: Data Ingestion & Cleaning

python -m src.data.load_data

Outputs:

data/processed/telco_churn_clean.parquet

Step 2: Model Training & Logging

python -m src.models.train_model

This trains multiple models (e.g., Logistic Regression, Random Forest), logs metrics to MLflow, and saves the best pipeline:

models/best_model_<name>.joblib


⸻

📈 6. MLflow UI

To explore experiments:

python -m mlflow ui --port 5001

Then open:

http://127.0.0.1:5001

You’ll see:
	•	Model runs (log_reg, random_forest)
	•	Metrics (F1, ROC-AUC, precision, recall)
	•	Saved models & artifacts

⸻

⚡ 7. Run the FastAPI Inference Service

After training completes and a model exists in models/:

Local run (no Docker)

uvicorn app.fastapi_app:app --reload

Endpoints

Endpoint	Description	URL
GET /health	Health check	http://127.0.0.1:8000/health
GET /docs	Swagger UI	http://127.0.0.1:8000/docs
POST /predict	Single record prediction	see below
POST /predict_batch	Batch prediction	see below

Example Request /predict

{
  "data": {
    "Contract": "Month-to-month",
    "Internet Service": "Fiber optic",
    "Monthly Charges": 80,
    "Tenure Months": 5
  }
}

Example Response

{
  "churn_prediction": "Yes",
  "churn_probability": 0.72
}


⸻

💻 8. Streamlit Dashboard

Run the interactive UI for quick testing and demo.

streamlit run app/streamlit_app.py

Then open:

http://localhost:8501

Features:
	•	Single prediction form (contract, internet service, etc.)
	•	Batch prediction via CSV upload
	•	Download results as CSV

If you see a ModuleNotFoundError: No module named 'src', ensure:
	•	You run Streamlit from the project root, not app/, or
	•	Add the root path (already handled in code).

⸻

🐳 9. Dockerized API Deployment

Build the Docker image

docker build -t telco-churn-api .

Run the container

docker run -p 8000:8000 telco-churn-api

Then access:
	•	API health: http://127.0.0.1:8000/health
	•	Docs: http://127.0.0.1:8000/docs

Commands summary

Action	Command
Build image	docker build -t telco-churn-api .
Run container	docker run -p 8000:8000 telco-churn-api
Stop all containers	docker stop $(docker ps -q)


⸻

🔁 10. Reproducibility & Best Practices

This project intentionally does not version:
	•	/data/raw/
	•	/data/processed/
	•	/models/
	•	/mlruns/

They are regenerated locally to mimic real MLOps workflows:

Code in Git, artifacts in local or cloud storage.

To reproduce from scratch:

python -m src.data.load_data
python -m src.models.train_model
uvicorn app.fastapi_app:app --reload


⸻

🧪 11. Typical Workflow Summary

Step	Command	Result
1. Load data	python -m src.data.load_data	Clean parquet ready
2. Train models	python -m src.models.train_model	Best pipeline saved
3. Run API	uvicorn app.fastapi_app:app --reload	/predict ready
4. Track runs	python -m mlflow ui --port 5001	MLflow dashboard
5. Launch Streamlit	streamlit run app/streamlit_app.py	Visual UI
6. Containerize	docker build -t telco-churn-api .	Portable API image


⸻

🧰 12. Troubleshooting

Issue	Fix
ModuleNotFoundError: No module named 'src'	Run Streamlit from project root
Cannot connect to Docker daemon	Open Docker Desktop first
MLflow UI not loading	Reinstall full mlflow (not skinny): pip install mlflow==3.6.0
Permission denied when running Docker	On macOS, restart Docker Desktop and retry


⸻

🚀 13. Future Enhancements
	•	✅ CI/CD pipeline via GitHub Actions
	•	Model registry & versioning
	•	Monitoring & drift detection
	•	Integration with a cloud-hosted MLflow tracking server



