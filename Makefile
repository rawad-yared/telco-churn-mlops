# Telco Churn MLOps Makefile


# Default environment variables
PYTHON := python
APP_PORT := 8000
STREAMLIT_PORT := 8501
MLFLOW_PORT := 5001


# Data & Model Pipeline


.PHONY: load_data train mlflow api streamlit docker-build docker-run clean help

load_data:
	@echo "🔹 Loading and cleaning Telco data..."
	$(PYTHON) -m src.data.load_data

train:
	@echo "🔹 Training models and logging to MLflow..."
	$(PYTHON) -m src.models.train_model

mlflow:
	@echo "🔹 Launching MLflow UI on port $(MLFLOW_PORT)..."
	$(PYTHON) -m mlflow ui --port $(MLFLOW_PORT)


# Run Applications


api:
	@echo "🚀 Starting FastAPI server..."
	uvicorn app.fastapi_app:app --host 0.0.0.0 --port $(APP_PORT) --reload

streamlit:
	@echo "📊 Launching Streamlit dashboard..."
	streamlit run app/streamlit_app.py --server.port $(STREAMLIT_PORT)


# Docker Commands


docker-build:
	@echo "🐳 Building Docker image 'telco-churn-api'..."
	docker build -t telco-churn-api .

docker-run:
	@echo "🐳 Running Docker container on port $(APP_PORT)..."
	docker run -p $(APP_PORT):8000 -v "$$(pwd)/models:/app/models" telco-churn-api


# Maintenance


clean:
	@echo "🧹 Cleaning processed data, models, and MLflow runs..."
	rm -rf data/processed/* models/* mlruns/* || true
	@echo "✅ Cleanup complete."

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make load_data        - Load and preprocess Telco data"
	@echo "  make train            - Train model(s) and log results to MLflow"
	@echo "  make mlflow           - Launch MLflow tracking UI"
	@echo "  make api              - Run FastAPI service locally"
	@echo "  make streamlit        - Launch Streamlit dashboard"
	@echo "  make docker-build     - Build Docker image for FastAPI API"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make clean            - Remove data/processed, models, and mlruns"
	@echo ""
	@echo "Example: make train && make api"
	@echo ""