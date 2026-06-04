# Model Card: Customer Review Rating Prediction System

## Model Overview

This project predicts customer review ratings (1–5 stars) using both review text and structured e-commerce metadata.

The final system combines:
- TF-IDF text representation
- TruncatedSVD dimensionality reduction
- HistGradientBoostingClassifier

The primary business goal is early identification of dissatisfied customers for proactive intervention.

# Intended Use

This model is designed to:
- Predict customer review ratings
- Identify potentially dissatisfied customers
- Support customer service intervention workflows
- Assist businesses in prioritizing customer follow-up

The model is intended as a decision-support tool rather than a fully automated decision-making system.



# Out-of-Scope Use

This model should NOT:
- Automatically suppress or modify customer reviews
- Replace human customer support decisions
- Be used as the sole basis for punitive business actions
- Be deployed without ongoing monitoring and retraining

The model predicts likely ratings but does not determine the true intent or emotional state of a customer.



# Inputs

## Text Features
- review_text

## Structured Features
- product_price
- seller_rating
- delivery_days
- product_age_months
- num_previous_reviews_by_user
- verified_purchase
- return_initiated
- helpful_votes
- discount_pct
- product_weight_kg
- image_count
- product_category

Additional structured features may be incorporated depending on deployment requirements.



# Output

The model predicts one of the following classes:
- Rating 1
- Rating 2
- Rating 3
- Rating 4
- Rating 5

The system also supports threshold-based identification of dissatisfied customers (ratings 1–2).



# Model Architecture

## Text Representation
- TF-IDF Vectorization
- max_features = 1000
- ngram_range = (1,1)
- min_df = 5

## Dimensionality Reduction
- TruncatedSVD
- n_components = 200

## Final Classifier
- HistGradientBoostingClassifier

The final hyperparameters were selected using RandomizedSearchCV with cross-validation.


# Performance Metrics

## Final Tuned Model
- Weighted F1 = 0.6055
- Macro F1 = 0.6005

## Dissatisfied Customer Detection
Threshold = 0.40 for combined probability of ratings 1–2

- Recall: 0.85
- Precision: 0.88

The tuned HistGradientBoosting model performed competitively with the best ensemble baseline while maintaining a simpler and more efficient architecture.



# Known Failure Modes

## Deceptive Reviews

The model struggles when review sentiment contradicts the assigned rating.

Example:
- Positive review text paired with a 1-star rating due to delivery or seller issues.



## Short or Uninformative Reviews

Very short reviews provide insufficient textual signal, forcing the model to rely more heavily on structured features.



## Adjacent Rating Confusion

The model occasionally confuses nearby rating classes:
- 1 vs 2
- 4 vs 5

This occurs when review language is subtle, ambiguous, or emotionally mixed.


## Conflicting Modalities

Cases where structured features contradict otherwise positive review text may reduce accuracy.

Example:
- Positive product feedback paired with extremely long delivery times.



# Ethical Considerations

The model may inherit biases present in the training data, including:
- customer language patterns
- seller-related imbalance
- product category imbalance

Predictions should always be interpreted with human oversight.

The model is intended to support customer experience improvement rather than manipulate customer feedback.



# Training Data Limitations

The training dataset contains approximately:
- 2,000 samples
- short customer reviews
- relatively limited vocabulary diversity

Because of this, the model may not generalize perfectly to significantly different products, domains, or customer populations without retraining.



# Monitoring Recommendations

Recommended production monitoring includes:

## Data Drift Monitoring
Monitor:
- vocabulary drift
- changes in review writing style
- shifts in structured feature distributions

Retrain the model if significant drift is detected.



## Performance Monitoring
Continuously track:
- Weighted F1
- Macro F1
- Recall for dissatisfied customers
- Precision for intervention triggers



## False Positive / False Negative Analysis
Regularly review:
- satisfied customers incorrectly flagged
- dissatisfied customers missed by the model

This helps identify recurring failure patterns.



## Bias Monitoring
Evaluate model performance across:
- product categories
- seller groups
- pricing ranges

to ensure fair and stable behavior.


# Deployment Notes

The final system is implemented as:
- modular Scikit-learn pipeline
- reusable preprocessing architecture
- deployable Streamlit application

The pipeline supports:
- training
- prediction
- threshold calibration
- future retraining workflows