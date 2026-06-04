import joblib
import pandas as pd
from pathlib import Path

# Load Model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)

# Sample Input
sample_data = pd.DataFrame([{
    "review_text": "The product was amazing and arrived quickly",
    "product_price": 49.99,
    "product_category": "electronics",
    "seller_rating": 4.5,
    "delivery_days": 2,
    "product_age_months": 12,
    "num_previous_reviews_by_user": 5,
    "verified_purchase": 1,
    "return_initiated": 0,
    "helpful_votes": 3,
    "discount_pct": 20,
    "product_weight_kg": 1.2,
    "image_count": 2,
    "return_initiated": 0
}])

# Prediction
prediction = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

# Output
print("Predicted rating:", prediction[0])
print("Probabilities:", probabilities[0])