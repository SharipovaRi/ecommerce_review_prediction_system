import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.final_pipeline_def import get_pipeline

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# Load Data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["rating"])
y = df["rating"]

# Fill missing text
X["review_text"] = X["review_text"].fillna("")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load Pipeline
model = get_pipeline()

# Train Model
model.fit(X_train, y_train)

# Save Final Model
joblib.dump(model, MODEL_PATH)

print("Model training complete and saved as model.pkl")