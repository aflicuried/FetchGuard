from fastapi import APIRouter
from pydantic import BaseModel
from typing import List


router = APIRouter()


class CTGSample(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    label: str
    probabilities: List[float]


@router.post("/predict", response_model=PredictionResponse)
def predict(sample: CTGSample) -> PredictionResponse:
    # Placeholder stub. Wire to real model in src/models/predict.py
    probabilities = [1.0, 0.0, 0.0]
    label = "Normal"
    return PredictionResponse(label=label, probabilities=probabilities)


