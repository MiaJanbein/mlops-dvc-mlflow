
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# 1. Load the downloaded model
model_path = Path("model")

model = mlflow.pyfunc.load_model(str(model_path))


# 2. Create the FastAPI application
app = FastAPI()


# 3. Define the expected input format
class PredictionInput(BaseModel):
    inputs: list[list[float]]


# 4. Create the only endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    predictions = model.predict(
        np.array(data.inputs)
    )

    return {
        "predictions": predictions.tolist()
    }
