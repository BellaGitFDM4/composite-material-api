from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3

# Initialize DB
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            density REAL,
            tensile_strength REAL,
            category TEXT,
            process TEXT,
            description TEXT,
            is_composite INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


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
    
    # Log to SQLite
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (density, tensile_strength, category, process, description, is_composite)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.density,
        data.tensile_strength,
        data.category,
        data.process,
        data.description,
        int(prediction)
    ))
    conn.commit()
    conn.close()

    return {"is_composite": bool(prediction)}

from fastapi import UploadFile, File
import io

@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(file.file.read()))
    predictions = model.predict(df)
    df["is_composite"] = predictions

    # Log each row to SQLite
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO predictions (density, tensile_strength, category, process, description, is_composite)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["density"],
            row["tensile_strength"],
            row["category"],
            row["process"],
            row["description"],
            int(row["is_composite"])
        ))

    conn.commit()
    conn.close()

    return df.to_dict(orient="records")

