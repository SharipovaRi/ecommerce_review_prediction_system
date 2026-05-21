import joblib
import pandas as pd
from pathlib import Path

# Load Model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# Load trained sklearn pipeline
def load_model():
    return joblib.load(MODEL_PATH)

# Single sample dictionary
def predict(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]


def predict_proba(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict_proba(df)[0]