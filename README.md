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
* Full-stack deployment using Railway (FastAPI backend + Streamlit frontend)

## Live Demo
### Streamlit App (UI):

https://e-commerce-review-prediction-system.up.railway.app/

### How to Use:

1. Open Streamlit UI link
2. Enter review details
3. Click "Predict Rating"
4. View prediction and probability breakdown
   
### FastAPI backend link (API):

https://backend-e-commerce-review-prediction-system.up.railway.app/docs

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
* Hyperparameter tuning performed using RandomizedSearchCV
* Threshold calibration for dissatisfied customer detection

### Pipeline
Fully modular sklearn Pipeline combining:

* ColumnTransformer
* Feature pipelines
* Final classifier

## System Architecture

The project follows a modular production-style machine learning architecture separating frontend, backend, inference pipeline, and deployment components.

![System Architecture](images/system_architecture.png)

### Architecture Components

#### Clients
Users interact with the system through the Streamlit interface by:

* Entering review text and product details
* Viewing predicted ratings
* Viewing dissatisfaction risk probabilities
* Viewing probability distributions

#### Frontend (Streamlit)

The Streamlit frontend provides:

* Interactive prediction interface
* Probability visualization
* Real-time communication with the backend API

#### Backend API (FastAPI)

The FastAPI backend handles:

* Request validation using Pydantic
* Data preprocessing and validation
* Feature engineering pipeline execution
* Model inference
* JSON response generation

#### ML Pipeline

The machine learning pipeline includes:

* TF-IDF vectorization
* TruncatedSVD dimensionality reduction
* Structured feature preprocessing
* HistGradientBoostingClassifier inference

The system combines text, numerical, and categorical features using Scikit-learn pipelines and column transformers.

#### Model & Artifacts

Saved artifacts include:

* Trained model (`model.pkl`)
* TF-IDF vectorizer
* TruncatedSVD transformer
* ColumnTransformer preprocessing pipeline
* Encoders and scalers

#### Outputs

The system produces:

* Predicted rating (1–5)
* Dissatisfaction probability score
* Full probability distribution
* Confidence score

#### Infrastructure & DevOps

The project includes:

* Docker containerization
* GitHub Actions CI/CD
* Automated testing with pytest
* Railway deployment

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
* Precision: 0.88
* Recall: 0.85

The model is effective at identifying dissatisfied customers, which is the key business objective.

