# 📡 Telco Churn MLOps Project

[![CI-CD](https://github.com/rawad-yared/telco-churn-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rawad-yared/telco-churn-mlops/actions/workflows/ci-cd.yml)

End-to-end churn prediction project using the **IBM Telco Customer Churn** dataset:

- Reproducible data & feature pipelines  
- Model training & evaluation with scikit-learn  
- Experiment tracking with MLflow  
- Containerized deployment with Docker  
- FastAPI API for online inference  
- Streamlit dashboard for interactive predictions  

⸻

🧭 1. Project Overview

A full end-to-end MLOps pipeline for predicting telecom customer churn using the IBM Telco Customer Churn dataset.

The goal is to classify whether a customer will churn based on contract type, services, tenure, billing patterns, and demographics.

🔧 Tech Stack:
	•	Python 3.12, pandas, scikit-learn
	•	MLflow for experiment tracking
	•	FastAPI for online inference
	•	Streamlit for dashboarding
	•	Docker for containerization
	•	GitHub Actions for CI/CD
	•	GitHub Container Registry (GHCR) for image hosting
	•	Render.com for cloud deployment

🔁 MLOps Pipeline (High-Level)

Git push → GitHub Actions → Train → Build → Test → Push Image → Render Deploy → Live API



---


## 🗂️ 2. Repository Structure


```bash
telco-churn-mlops/
├─ app/
│  ├─ fastapi_app.py         # FastAPI serving API
│  └─ streamlit_app.py       # Optional Streamlit UI
├─ src/
│  ├─ data/
│  │  └─ load_data.py        # Load + clean dataset
│  ├─ features/
│  │  └─ build_features.py
│  └─ models/
│     ├─ train_model.py      # Model training, MLflow logging
│     └─ predict_model.py     # Schema-aligned inference pipeline
├─ data/
│  ├─ raw/
│  │  └─ Telco_customer_churn.xlsx   # Dataset (tracked in Git)
│  └─ processed/                      # Generated
├─ models/                             # Generated (artifacts)
├─ mlruns/                             # Local MLflow logs
├─ Dockerfile
├─ requirements.txt
├─ Makefile
└─ README.md


## ⚙️ 3. Setup Instructions

### 🪄 Clone the repository

```bash
git clone https://github.com/rawad-yared/telco-churn-mlops.git
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

The dataset is already included and tracked:

data/raw/Telco_customer_churn.xlsx


⸻

🧩 5. Automated MLOps Pipeline (CI/CD)

Once you clone the repository and install dependencies, you do not need to run the pipeline manually.
This project uses a full CI/CD workflow with:
	•	GitHub Actions (CI)
	•	GitHub Container Registry (GHCR)
	•	Render (CD)

Every time you push to the main branch, the entire workflow runs end-to-end.

Below is how to set it up.

⸻

🔧 Step 1 — Create TWO Render Services

You must create two Web Services on Render:

⸻

1️⃣ FastAPI Production API
	•	Render → New → Web Service
	•	Choose “Deploy an existing image”
	•	Use image (will be created automatically on first push):

ghcr.io/<your-username>/telco-churn-mlops:latest


	•	Start Command:

uvicorn app.fastapi_app:app --host 0.0.0.0 --port $PORT


	•	Save the service → copy the Service ID

⸻

2️⃣ Streamlit Dashboard (UI)
	•	Create a second Web Service
	•	Use the same GHCR image
	•	Start Command: leave it empty

⸻

🔐 Step 2 — Add GitHub Secrets (Required for CI/CD)

Go to:
GitHub → Repo → Settings → Secrets → Actions

Create these secrets:

Secret	What it is
RENDER_API_KEY	From Render → Account → API Keys
RENDER_SERVICE_ID	Streamlit Service ID
RENDER_FASTAPI_SERVICE_ID	FASTAPI Service ID

These allow GitHub Actions to deploy automatically after building the image.

⸻

🚀 Step 3 — Push to GitHub (CI/CD Runs Automatically)

Once the secrets and Render services are configured, you never run the pipeline manually again.

Simply do:

git add .
git commit -m "update"
git push

GitHub Actions will automatically:

CI Phase
	1.	Install dependencies
	2.	Load Telco dataset
	3.	Train Logistic Regression + Random Forest
	4.	Log metrics with MLflow
	5.	Save the best model
	6.	Build Docker image
	7.	Health-check the image locally in CI
	8.	Push image → GHCR

CD Phase
	9.	Trigger Render deploy for FastAPI
	10.	(Optional) Trigger Render deploy for Streamlit
	11.	Render pulls the new image
	12.	Your updated API + Dashboard are live automatically

⸻

🌐 Step 4 — Visit Your Live Services

After the first successful push:

FastAPI (production)

https://<your-fastapi-service>.onrender.com/docs

Streamlit dashboard

https://<your-streamlit-service>.onrender.com

These update automatically on every push

