import os
import xgboost as xgb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "historical_transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_xgboost.json")

data = pd.read_csv(DATA_PATH)
X = data[["amount", "is_high_value"]]
y = data["is_fraud"]

model = xgb.XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X, y)
model.save_model(MODEL_PATH)
