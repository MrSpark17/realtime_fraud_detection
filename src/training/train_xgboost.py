import xgboost as xgb
import pandas as pd

data = pd.read_csv("historical_transactions.csv")
X = data[["amount", "is_high_value"]]
y = data["is_fraud"]

model = xgb.XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X, y)
model.save_model("fraud_model.json")