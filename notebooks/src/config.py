from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# Cross-validation setup

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Feature names

text_feature = "review_text"

numeric_features = [
    "product_price",
    "seller_rating",
    "delivery_days",
    "product_age_months",
    "num_previous_reviews_by_user",
    "verified_purchase",
    "return_initiated",
    "helpful_votes",
    "discount_pct",
    "product_weight_kg",
    "image_count"
]

# Target and feature setup
target = "rating"

text_feature = "review_text"

numeric_features = [
    "product_price",
    "seller_rating",
    "delivery_days",
    "product_age_months",
    "num_previous_reviews_by_user",
    "verified_purchase",
    "return_initiated",
    "helpful_votes",
    "discount_pct",
    "product_weight_kg",
    "image_count"
]

categorical_features = ["product_category"]

structured_features = numeric_features + categorical_features


# Pipelines

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.95
    )),
    ("svd", TruncatedSVD(n_components=200, random_state=42))
])


# Preprocessors

structured_only_preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

text_only_preprocessor = ColumnTransformer([
    ("text", text_pipeline, text_feature)
])

combined_preprocessor = ColumnTransformer([
    ("text", text_pipeline, text_feature),
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])