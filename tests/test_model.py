from app.main import model
import pandas as pd

def test_model_prediction():
    sample = pd.DataFrame([{
        "review_text": "Good product",
        "delivery_days": 2,
        "seller_rating": 4.5,
        "product_price": 20.0,
        "product_age_months": 3,
        "return_initiated": 0,
        "verified_purchase": 1,
        "num_previous_reviews_by_user": 3,
        "helpful_votes": 5,
        "discount_pct": 0.1,
        "product_weight_kg": 1.0,
        "image_count": 1,
        "product_category": 0
    }])

    pred = model.predict(sample)

    assert pred[0] in [1, 2, 3, 4, 5]