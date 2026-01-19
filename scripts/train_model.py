import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.common.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("train_model")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = get_config()
model_cfg = cfg["models"]

DATA_PATH = os.path.join(BASE_DIR, "data", "historical_transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, model_cfg["xgboost_path"])

def load_and_prepare_data(data_path):
    """Load and prepare data for training."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    
    logger.info("Data shape: %s", df.shape)
    logger.info("Fraud cases: %d (%.2f%%)", df["is_fraud"].sum(), df["is_fraud"].mean() * 100)
    
    # Feature engineering
    df["location_encoded"] = LabelEncoder().fit_transform(df["location"])
    
    # Select features and target
    feature_cols = ["amount", "is_high_value", "location_encoded"]
    X = df[feature_cols]
    y = df["is_fraud"]
    
    return X, y

def train_model(X, y):
    """Train XGBoost model."""
    logger.info("Splitting data into train/test (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info("ROC-AUC Score: %.4f", roc_auc)
    
    logger.info("\nClassification Report:")
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\n%s", cm)
    
    return model

def save_model(model, model_path):
    """Save trained model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info("✓ Model saved to %s", model_path)

if __name__ == "__main__":
    logger.info("Starting model training pipeline...")
    
    # Load data
    X, y = load_and_prepare_data(DATA_PATH)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model, MODEL_PATH)
    
    logger.info("\n✓ Model training complete!")
