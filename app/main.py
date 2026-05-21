from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from pathlib import Path

# Load trained model
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# Basic model monitoring
LOG_PATH = Path("logs/predictions.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# FastAPI app 
app = FastAPI(
    title="Customer Review Rating Prediction API",
    description="Predicts customer review ratings and dissatisfaction probability using text + structured features.",
    version="1.0"
)


# Input schema
class ReviewInput(BaseModel):
    review_text: str
    delivery_days: int
    seller_rating: float
    product_price: float
    product_age_months: int
    return_initiated: int
    num_previous_reviews_by_user: int
    helpful_votes: int
    discount_pct: float
    product_weight_kg: float
    image_count: int
    product_category: str

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Rating Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict_rating(data: ReviewInput):

    # Convert input into dataframe
    input_df = pd.DataFrame([{
        "review_text": data.review_text,
        "delivery_days": data.delivery_days,
        "seller_rating": data.seller_rating,
        "product_price": data.product_price,
        "product_age_months": data.product_age_months,
        "return_initiated": data.return_initiated,
        "num_previous_reviews_by_user": data.num_previous_reviews_by_user,
        "helpful_votes": data.helpful_votes,
        "discount_pct": data.discount_pct,
        "product_weight_kg": data.product_weight_kg,
        "image_count": data.image_count,
        "product_category": data.product_category
    }])


    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Class order
    classes = model.named_steps["model"].classes_

    # Convert to dictionary
    probability_dict = {
        int(classes[i]): float(probabilities[i])
        for i in range(len(classes))
    }

    # Dissatisfied probability: Ratings 1 and 2
    dissatisfied_probability = (
        probability_dict.get(1, 0)
        + probability_dict.get(2, 0)
    )
    
    # Logging 
    logging.info({
        "input": data.model_dump(),
        "prediction": int(prediction),
        "confidence": float(max(probabilities)),
        "dissatisfied_probability": float(dissatisfied_probability)
    })


    # Return response
    return {
        "predicted_rating": int(prediction),
        "dissatisfied_probability": round(dissatisfied_probability, 4),
        "class_probabilities": probability_dict
    }