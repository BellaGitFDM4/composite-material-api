from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("material_classifier.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class MaterialInput(BaseModel):
    density: float
    tensile_strength: float
    category: str
    process: str
    description: str

@app.post("/predict")
def predict_material(data: MaterialInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"is_composite": bool(prediction)}

from fastapi import UploadFile, File
import io

@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(file.file.read()))
    predictions = model.predict(df)
    df["is_composite"] = predictions
    return df.to_dict(orient="records")
