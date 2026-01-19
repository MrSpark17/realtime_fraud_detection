"""
Quick demo script to show fraud detection capability.
Trains model, generates sample transactions, and demonstrates predictions.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import IsolationForest

from src.common.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("quick_demo")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = get_config()
model_cfg = cfg["models"]

DATA_PATH = os.path.join(BASE_DIR, "data", "historical_transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, model_cfg["xgboost_path"])

def main():
    logger.info("=" * 70)
    logger.info("Real-Time Fraud Detection System - Quick Demo")
    logger.info("=" * 70)
    
    # Load training data
    logger.info("\n1. Loading historical transaction data...")
    if not os.path.exists(DATA_PATH):
        logger.error("Data file not found. Run: python scripts/generate_sample_data.py")
        return
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"   ✓ Loaded {len(df)} transactions")
    logger.info(f"   ✓ Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    
    # Load or train model
    logger.info("\n2. Loading XGBoost model...")
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        logger.info(f"   ✓ Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"   ✗ Model not found at {MODEL_PATH}")
        logger.info("   Run: python scripts/train_model.py")
        return
    
    # Prepare test data
    logger.info("\n3. Preparing test data...")
    df["location_encoded"] = LabelEncoder().fit_transform(df["location"])
    feature_cols = ["amount", "is_high_value", "location_encoded"]
    X = df[feature_cols]
    
    # Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=float(model_cfg["isolation_contamination"]),
        random_state=int(model_cfg["random_state"])
    )
    iso_forest.fit(df[["amount"]])
    
    # Get predictions
    logger.info("\n4. Running hybrid fraud detection on sample transactions...")
    fraud_proba = model.predict_proba(X)[:, 1]
    anomaly_scores = iso_forest.decision_function(df[["amount"]])
    is_anomaly = iso_forest.predict(df[["amount"]])
    
    # Apply ensemble rule
    alerts = df[
        (fraud_proba > 0.8) & 
        (is_anomaly == -1)
    ].copy()
    
    alerts["fraud_probability"] = fraud_proba[
        (fraud_proba > 0.8) & 
        (is_anomaly == -1)
    ]
    alerts["anomaly_score"] = anomaly_scores[
        (fraud_proba > 0.8) & 
        (is_anomaly == -1)
    ]
    
    # Display results
    logger.info(f"\n5. Results:")
    logger.info(f"   ✓ Total transactions analyzed: {len(df)}")
    logger.info(f"   ✓ High-risk alerts triggered: {len(alerts)}")
    logger.info(f"   ✓ Alert rate: {len(alerts)/len(df)*100:.2f}%")
    
    if len(alerts) > 0:
        logger.info(f"\n   Top 5 High-Risk Transactions:")
        logger.info("   " + "-" * 60)
        for idx, row in alerts.nlargest(5, "fraud_probability").iterrows():
            logger.info(
                f"   TX: {row['transaction_id']} | "
                f"Amount: ${row['amount']:.2f} | "
                f"Fraud Prob: {row['fraud_probability']:.2%}"
            )
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Demo complete! System is working as expected.")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
