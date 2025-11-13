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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make project root (and src) importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose 8000 (local + CI + Render-compatible)
EXPOSE 8000

# Start Streamlit when the container runs
# If $PORT is set (Render), use it; otherwise default to 8000 (for local + CI)
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port ${PORT:-8000} --server.address 0.0.0.0"]