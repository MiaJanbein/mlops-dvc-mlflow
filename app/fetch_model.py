
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


# 1. Connect to the DagsHub MLflow server
tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()


# 2. Find the latest model in our experiment

# Find the latest model in our experiment
models = client.search_logged_models(
    experiment_ids=["0"],
    max_results=1,
    order_by=[
        {"field_name": "creation_time", "ascending": False}
    ],
)


if not models:
    raise RuntimeError("No logged models found in experiment 0.")

latest_model = models[0]

print(f"Latest model found: {latest_model.name}")
print(f"Model ID: {latest_model.model_id}")


# 3. Download the model files from DagsHub
model_dir = Path("model")

mlflow.artifacts.download_artifacts(
    artifact_uri=latest_model.artifact_location,
    dst_path=str(model_dir),
)

print(f"Model downloaded successfully to: {model_dir}")
