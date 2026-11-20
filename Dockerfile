# syntax=docker/dockerfile:1.7
# Use Python 3.12
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Prevent Python from creating unnecessary files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Download the latest model from DagsHub during image build
RUN --mount=type=secret,id=dagshub_token \
    MLFLOW_TRACKING_URI="https://dagshub.com/MiaJanbein/mlops-dvc-mlflow.mlflow" \
    MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_token)" \
    python app/fetch_model.py

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
