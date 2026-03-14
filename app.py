from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

app = FastAPI()

model = joblib.load("model.pkl")

class WineFeatures(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: WineFeatures):
    prediction = model.predict([data.features])
    return {
        "name": "Keshavi Ragipani",
        "roll_no": "2022BCS0122",
        "wine_quality": int(prediction[0])
    }