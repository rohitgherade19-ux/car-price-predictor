"""
train.py — Preprocessing + Linear Regression training pipeline
Run once: python backend/train.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'quikr_car.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')

# ── STEP 1: Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# ── STEP 2: Drop untrainable rows ─────────────────────────────────────
df = df[df['Price'] != 'Ask For Price'].copy()
df = df[df['year'].astype(str).str.fullmatch(r'\d{4}')].copy()

# ── STEP 3: Type casting ──────────────────────────────────────────────
df['year']  = df['year'].astype(int)
df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False).astype(int)

df['kms_driven'] = (
    df['kms_driven'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace(' kms', '', regex=False)
)
df.loc[df['kms_driven'] == 'nan', 'kms_driven'] = np.nan
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

df['company'] = df['company'].astype(str).str.strip()
df['company'] = df['company'].replace({'MARUTI': 'Maruti'})

# ── STEP 4: Drop junk companies ───────────────────────────────────────
junk = {'selling', 'urjent', 'i', 'used', 'commercial', 'well', 'all', 'any', 'no'}
real_short = {'tata', 'bmw', 'kia', 'fiat', 'mini'}
mask = df['company'].str.lower().isin(junk) | (
    (df['company'].str.len() <= 3) & (~df['company'].str.lower().isin(real_short))
)
df = df[~mask].copy()

# ── STEP 5: Drop nulls ────────────────────────────────────────────────
df = df.dropna(subset=['fuel_type', 'kms_driven']).copy()
df['kms_driven'] = df['kms_driven'].astype(int)

# ── STEP 6: Duplicates ────────────────────────────────────────────────
df = df.drop_duplicates().copy()

# ── STEP 7: Outliers ──────────────────────────────────────────────────
df = df[(df['Price'] < 6_000_000) &
        (df['kms_driven'] < 500_000) &
        (df['kms_driven'] > 100)].copy()

# ── STEP 8: Feature engineering ───────────────────────────────────────
df['name']    = df['name'].astype(str).apply(lambda x: ' '.join(x.split()[:3]))
df['car_age'] = 2026 - df['year']
df = df.reset_index(drop=True)
print(f"After cleaning: {df.shape}")

# ── STEP 9: Encode ────────────────────────────────────────────────────
X = df.drop(columns=['Price', 'year'])
y = df['Price']

X_encoded = pd.get_dummies(X, columns=['name', 'company', 'fuel_type'], drop_first=False)

# ── STEP 10: Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ── STEP 11: Scale ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── STEP 12: Train ────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
print(f"R2 : {r2_score(y_test, preds):.3f}")
print(f"MAE: Rs.{mean_absolute_error(y_test, preds):,.0f}")

# ── STEP 13: Save all artifacts ───────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, 'columns.pkl'), 'wb') as f:
    pickle.dump(list(X_encoded.columns), f)

meta = {
    'companies':  sorted(df['company'].unique().tolist()),
    'names':      sorted(df['name'].unique().tolist()),
    'fuel_types': sorted(df['fuel_type'].unique().tolist()),
    'year_min':   int(df['year'].min()),
    'year_max':   int(df['year'].max()),
}
with open(os.path.join(MODEL_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Saved: model.pkl, scaler.pkl, columns.pkl, meta.pkl")
