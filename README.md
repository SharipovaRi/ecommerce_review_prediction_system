# Customer Review Rating Prediction System
## Overview
This project is an end-to-end Machine Learning system that predicts customer review ratings (1–5 ratings) and identifies dissatisfaction risk based on both review text and structured e-commerce data. 
The system is designed with a modular ML pipeline separating data processing, model training, and deployment components, making it suitable as a production-style ML application.

## Disclaimer 
Originally, this project was developed as part of a graduate-level Machine Learning course final group project. I later extended and enhanced into a production-style machine learning application.

Predictions are probabilistic and should not be used as the sole basis for business decisions.

## Key Features 

* Natural language processing of customer reviews
* Multi-class rating prediction (ratings 1–5)
* Dissatisfaction risk scoring (ratings 1–2)
* Probability distribution visualization
* Real-time inference API (FastAPI)
* Interactive UI (Streamlit)
* Containerized deployment (Docker-ready)
* Reproducible ML pipeline
* Automated testing (pytest + CI/CD)
* Basic model logging for monitoring
* Deployed as a full-stack application using Railway

## Live Demo
### Streamlit App (UI):

https://e-commerce-review-prediction.up.railway.app/

### How to Use:

1. Open Streamlit UI link
2. Enter review details
3. Click "Predict Rating"
4. View prediction and probability breakdown
   
### FastAPI backend link (API):

https://ecommercereviewpredictionsystem-production.up.railway.app/docs

## Problem Statement
Online retailers often struggle to identify dissatisfied customers early. This system enables proactive customer support and business intervention. It predicts:

* Rating (1–5)
* Probability of dissatisfaction (ratings 1–2)
* Full probability distribution across ratings 

## Machine Learning Approach

### Feature Engineering
* Text Features: TF-IDF (vectorization of review text) and SVD (dimensionality reduction)
* Numerical Features: StandardScaler 
* Categorical Features: OneHotEncoding

### Model
* HistGradientBoostingClassifier (Scikit-learn)
* Hyperparameter tuning performed 

### Pipeline
Fully modular sklearn Pipeline combining:

* ColumnTransformer
* Feature pipelines
* Final classifier

## Tech Stack

### Machine Learning

* Scikit-learn
* TF-IDF and TruncatedSVD
* HistGradientBoostingClassifier

### Backend

* FastAPI
* Pydantic

### Frontend

* Streamlit

### DevOps

* Docker (containerized)
* GitHub Actions (CI/CD)

### Testing

* Pytest
* HTTPX (API testing)

## Running the Project Locally
1. Clone repository
   
`git clone https://github.com/SharipovaRi/ecommerce_review_prediction_system.git
`

`cd ecommerce_review_prediction_system
`

2. Install dependencies
   
`pip install -r requirements.txt
`

3. Run FastAPI

`uvicorn app.main:app --reload
`

4. Run Streamlit app

`streamlit run streamlit_app.py
`

## Running Tests

`python -m pytest tests -v   
`
## CI/CD Pipeline
This project uses GitHub Actions to:

* Install dependencies
* Run unit tests
* Validate API endpoints
* Ensure reproducibility

## Docker
The application is containerized to ensure reproducible environments across development and deployment.

`docker build -t review-model .
docker run -p 8000:8000 review-model
`

## Monitoring and Logging
The API includes logging to track prediction requests, model outputs and scores. 
Stored locally at: logs/predictions.log

## Model Performance 
The final model (HistGradientBoostingClassifier with TF-IDF + SVD features) achieved:

* Weighted F1-score: 0.61
* Macro F1-score: 0.60
* Accuracy: ~0.61

Performance is consistent with a small-to-medium sized dataset and a multi-class classification setup with mixed text and structured features.

## Dissatisfied Customer Detection (Ratings 1–2)
* Precision: 0.92
* Recall: 0.83

The model is effective at identifying dissatisfied customers, which is the key business objective.

