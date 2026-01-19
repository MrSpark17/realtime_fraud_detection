import os
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, LongType

import xgboost as xgb
from sklearn.ensemble import IsolationForest
import joblib
import pandas as pd

from src.common.config import get_config

# Load config
cfg = get_config()
kafka_cfg = cfg["kafka"]
stream_cfg = cfg["streaming"]
model_cfg = cfg["models"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("streaming_job")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", model_cfg["xgboost_path"])

# Spark session
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Define schema
schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("merchant_id", IntegerType()),
    StructField("location", StringType()),
    StructField("timestamp", LongType())
])

# Read from Kafka
raw_df = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
         .option("subscribe", kafka_cfg["transactions_topic"])
         .option("startingOffsets", "latest")
         .load()
)

transactions = (
    raw_df
    .select(from_json(col("value").cast("string"), schema).alias("data"))
    .select("data.*")
)

# Feature engineering
features_df = transactions.withColumn(
    "is_high_value",
    when(col("amount") > float(stream_cfg["high_value_threshold"]), 1).otherwise(0)
)

# Load models once (on driver)
logger.info("Loading XGBoost model from %s", MODEL_PATH)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_PATH)

logger.info("Initializing IsolationForest")
iso_forest = IsolationForest(
    contamination=float(model_cfg["isolation_contamination"]),
    random_state=int(model_cfg["random_state"])
)

def process_batch(batch_df, batch_id: int):
    logger.info("Processing batch_id=%s", batch_id)
    pdf = batch_df.select("transaction_id", "amount", "is_high_value").toPandas()

    if pdf.empty:
        logger.info("Empty batch; skipping")
        return

    # Anomaly scores
    iso_forest.fit(pdf[["amount"]])
    pdf["anomaly_score"] = iso_forest.decision_function(pdf[["amount"]])
    pdf["is_anomaly"] = iso_forest.predict(pdf[["amount"]])

    # XGBoost features
    X = pdf[["amount", "is_high_value"]]
    fraud_proba = xgb_model.predict_proba(X)[:, 1]
    pdf["fraud_probability"] = fraud_proba

    # Hybrid rule: high probability + anomaly
    alerts = pdf[
        (pdf["fraud_probability"] > 0.8) &
        (pdf["is_anomaly"] == -1)
    ]

    if not alerts.empty:
        logger.warning("Fraud alerts:\n%s", alerts.to_string(index=False))
    else:
        logger.info("No fraud alerts in this batch.")

query = (
    features_df.writeStream
    .foreachBatch(process_batch)
    .outputMode("update")
    .start()
)

query.awaitTermination()
