import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)


# Page configuration
st.set_page_config(
    page_title="Customer Review Rating Predictor",
    layout="centered"
)

# Title
st.title("Customer Review Rating Prediction System")

st.write("""
This application predicts customer review ratings and dissatisfaction risk
using review text and structured e-commerce features.
""")

# Sidebar
st.sidebar.header("Additional Information")

st.sidebar.info("""
Features Used:
- Review text
- Delivery performance
- Seller quality
- Product metadata

Built With:
- Streamlit
- FastAPI
- Scikit-learn
- TF-IDF + SVD + HistGradientBoostingClassifier
""")

# Example reviews
example_reviews = {
    "Positive Review":
        "Excellent product and very fast delivery. Highly recommend!",

    "Neutral Review":
        "The product was okay and worked as expected.",

    "Negative Review":
        "Terrible quality. Arrived broken and I was very disappointed."
}

selected_example = st.selectbox(
    "Choose Example Review",
    list(example_reviews.keys())
)

# Review input
review_text = st.text_area(
    "Review Text",
    value=example_reviews[selected_example],
    height=120
)

st.subheader("Product & Purchase Information")

# Two-column layout
col1, col2 = st.columns(2)

with col1:

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0,
        max_value=60,
        value=2
    )

    seller_rating = st.number_input(
        "Seller Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=49.99
    )

    product_age_months = st.number_input(
        "Product Age (Months)",
        min_value=0,
        value=6
    )

    return_initiated = st.selectbox(
        "Return Initiated",
        [0, 1]
    )

with col2:

    num_previous_reviews_by_user = st.number_input(
        "Previous Reviews by User",
        min_value=0,
        value=3
    )

    helpful_votes = st.number_input(
        "Helpful Votes",
        min_value=0,
        value=5
    )

    discount_pct = st.number_input(
        "Discount Percentage",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

    product_weight_kg = st.number_input(
        "Product Weight (kg)",
        min_value=0.0,
        value=1.0
    )

    image_count = st.number_input(
        "Image Count",
        min_value=0,
        value=1
    )

# Product category
product_category = st.selectbox(
    "Product Category",
    [
        "Electronics",
        "Fashion",
        "Home",
        "Beauty",
        "Sports",
        "Books",
        "Toys"
    ]
)

# Prediction button
if st.button("Predict Rating"):

    # Create dataframe
    input_df = pd.DataFrame([{
        "review_text": review_text,
        "delivery_days": delivery_days,
        "seller_rating": seller_rating,
        "product_price": product_price,
        "product_age_months": product_age_months,
        "return_initiated": return_initiated,
        "num_previous_reviews_by_user":
            num_previous_reviews_by_user,
        "helpful_votes": helpful_votes,
        "discount_pct": discount_pct,
        "product_weight_kg": product_weight_kg,
        "image_count": image_count,
        "product_category": product_category
    }])

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Confidence score
    confidence = max(probabilities)

    # Dissatisfied probability
    dissatisfied_probability = (
        probabilities[0] + probabilities[1]
    )

    st.divider()

    # Results
    st.subheader("Prediction Results")

    st.success(
        f"Predicted Customer Rating: {prediction}"
    )

    st.write(
        f"Model Confidence: {confidence:.2%}"
    )

    # Dissatisfaction section
    st.subheader("Dissatisfaction Risk")

    st.progress(float(dissatisfied_probability))

    if dissatisfied_probability > 0.7:
        st.error("High dissatisfaction risk detected.")

    elif dissatisfied_probability > 0.4:
        st.warning("Moderate dissatisfaction risk detected.")

    else:
        st.success("Low dissatisfaction risk detected.")

    st.write(
        f"Dissatisfaction Probability: "
        f"{dissatisfied_probability:.2%}"
    )

    # Probability chart
    st.subheader("Rating Probability Distribution")

    probability_df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Probability": probabilities
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        probability_df["Rating"].astype(str),
        probability_df["Probability"]
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability by Rating")

    # Keep labels horizontal
    plt.xticks(rotation=0)

    # Add values above bars
    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    # Detailed probabilities
    st.subheader("Detailed Probabilities")

    detailed_df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Probability (%)":
            [round(p * 100, 2) for p in probabilities]
    })

    st.dataframe(
        detailed_df,
        use_container_width=True
    )

    # Prediction insights
    st.subheader("Prediction Insights")

    insights = []

    # Delivery impact
    if delivery_days > 7:
        insights.append(
            "Long delivery time may negatively affect customer satisfaction."
        )

    elif delivery_days <= 2:
        insights.append(
            "Fast delivery positively influenced the prediction."
        )

    # Seller rating impact
    if seller_rating >= 4.5:
        insights.append(
            "High seller rating positively influenced the prediction."
        )

    elif seller_rating < 3.5:
        insights.append(
            "Low seller rating increased dissatisfaction risk."
        )

    # Return impact
    if return_initiated == 1:
        insights.append(
            "Return initiation increased dissatisfaction probability."
        )

    # Discount impact
    if discount_pct >= 0.4:
        insights.append(
            "Large discount may have positively influenced customer perception."
        )

    # Helpful votes impact
    if helpful_votes >= 10:
        insights.append(
            "High helpful vote count suggests strong customer engagement."
        )

    # Review sentiment cues
    negative_words = [
        "broken",
        "terrible",
        "bad",
        "awful",
        "poor",
        "disappointed"
    ]

    positive_words = [
        "excellent",
        "great",
        "perfect",
        "amazing",
        "fast",
        "recommend"
    ]

    review_lower = review_text.lower()

    if any(word in review_lower for word in negative_words):
        insights.append(
            "Negative wording in the review text contributed to lower rating prediction."
        )

    if any(word in review_lower for word in positive_words):
        insights.append(
            "Positive wording in the review text contributed to higher rating prediction."
        )

    # Display insights
    if insights:

        for insight in insights:
            st.write(f"- {insight}")

    else:
        st.write(
            "No strong contributing factors were detected for this prediction."
        )

# Footer
st.divider()

st.caption("""
Disclaimer:
This Customer Review Rating Prediction System was originally created
as part of a graduate Machine Learning course and later expanded
into an end-to-end ML system with deployment, testing,
and interactive visualization.

Caution:
Predictions are probabilistic estimates generated by a machine learning
model and may not always reflect real-world outcomes.
""")