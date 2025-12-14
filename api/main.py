from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    probability = model.predict_proba([features]).max()
    return {
        "prediction": int(prediction[0]),
        "confidence": float(probability)
    }
