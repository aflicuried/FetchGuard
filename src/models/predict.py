from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np


CTG_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathologic"}


def load_model(model_path: Path):
    return joblib.load(model_path)


def predict_proba(model, features: List[float]) -> Tuple[str, List[float]]:
    x = np.asarray(features, dtype=float).reshape(1, -1)
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        probs = proba(x)[0].tolist()
        # Map predicted class index 0..2 to labels 1..3 if needed
        pred_idx = int(np.argmax(probs))
        label = [CTG_LABELS[i] for i in sorted(CTG_LABELS.keys())][pred_idx]
        return label, probs
    # Fallback for models without predict_proba
    y = model.predict(x)[0]
    label = CTG_LABELS.get(int(y), str(y))
    return label, [0.0, 0.0, 0.0]


