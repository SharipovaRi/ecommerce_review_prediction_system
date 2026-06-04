from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

# Feature Definitions
TEXT_FEATURE = "review_text"

NUMERIC_FEATURES = [
    "product_price",
    "seller_rating",
    "delivery_days",
    "product_age_months",
    "num_previous_reviews_by_user",
    "helpful_votes",
    "discount_pct",
    "product_weight_kg",
    "image_count"
]

CATEGORICAL_FEATURES = ["product_category"]


# Text Pipeline with the best settings
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),   
        min_df=5,
        stop_words="english"
    )),
    ("svd", TruncatedSVD(
        n_components=200,
        random_state=42
    ))
])


# Numeric Pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Pipeline

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# Preprocessor
preprocessor = ColumnTransformer([
    ("text", text_pipeline, TEXT_FEATURE),
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])


# Final best model
model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,
    max_depth=5,
    max_leaf_nodes=15,
    l2_regularization=0.01,
    random_state=42
)


# Final Pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Helper to get pipeline: Returns the trained pipeline object (for training or inference)
def get_pipeline():
    return final_pipeline