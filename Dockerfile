# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make src importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI when the container runs
CMD ["sh", "-c", "uvicorn app.fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}"]