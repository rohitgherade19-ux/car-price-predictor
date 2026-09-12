"""
app.py — FastAPI backend (Linear Regression version)
  POST /predict        → takes car details, returns predicted price
  GET  /history        → returns last 20 predictions from SQLite
  GET  /meta           → returns dropdown options for the frontend
"""
import os, pickle, sqlite3, datetime
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'model')
FRONTEND  = os.path.join(BASE, '..', 'frontend')
DB_PATH   = os.path.join(BASE, '..', 'data', 'predictions.db')

# ── Load model artifacts ──────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'model.pkl'),   'rb') as f: model   = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'columns.pkl'), 'rb') as f: COLUMNS = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'meta.pkl'),    'rb') as f: META    = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'),  'rb') as f: scaler  = pickle.load(f)

# ── SQLite setup ──────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            company     TEXT,
            year        INTEGER,
            kms_driven  INTEGER,
            fuel_type   TEXT,
            predicted   INTEGER,
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="Car Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────
class CarInput(BaseModel):
    name:       str
    company:    str
    year:       int
    kms_driven: int
    fuel_type:  str

# ── /meta — send dropdown options to frontend ─────────────────────────
@app.get("/meta")
def get_meta():
    return META

# ── /predict ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(car: CarInput):
    # 1. Build a single-row dataframe with the same features as training
    car_age = 2026 - car.year
    input_df = pd.DataFrame([{
        'name':       ' '.join(car.name.split()[:3]),
        'company':    car.company,
        'kms_driven': car.kms_driven,
        'fuel_type':  car.fuel_type,
        'car_age':    car_age,
    }])

    # 2. One-hot encode — must produce the exact same columns as training
    input_encoded = pd.get_dummies(input_df, columns=['name', 'company', 'fuel_type'])

    # 3. Reindex to match training columns (adds 0 for unseen categories)
    input_encoded = input_encoded.reindex(columns=COLUMNS, fill_value=0)

    # 4. Scale using the same scaler fitted during training
    input_scaled = scaler.transform(input_encoded)

    # 5. Predict — clamp to 0 so negative predictions don't show
    predicted_price = max(0, int(model.predict(input_scaled)[0]))

    # 6. Log to SQLite
    conn = get_db()
    conn.execute("""
        INSERT INTO predictions (name, company, year, kms_driven, fuel_type, predicted, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (car.name, car.company, car.year, car.kms_driven,
          car.fuel_type, predicted_price,
          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

    return {
        "predicted_price": predicted_price,
        "formatted":       f"₹{predicted_price:,}"
    }

# ── /history — last 20 predictions ───────────────────────────────────
@app.get("/history")
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Serve frontend ────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND, "index.html"))
