from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import *
import xgboost as xgb
from sklearn.ensemble import IsolationForest

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("merchant_id", IntegerType()),
    StructField("location", StringType()),
    StructField("timestamp", LongType())
])

df = spark.readStream.format("kafka")         .option("kafka.bootstrap.servers", "localhost:9092")         .option("subscribe", "transactions").load()

transactions = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

features = transactions.withColumn("is_high_value", when(col("amount") > 2000, 1).otherwise(0))

iso_forest = IsolationForest(contamination=0.02, random_state=42)
model = xgb.XGBClassifier()
model.load_model("fraud_model.json")

def process_batch(batch_df, batch_id):
    pdf = batch_df.select("amount", "is_high_value").toPandas()
    if pdf.empty:
        return
    pdf["anomaly"] = iso_forest.fit_predict(pdf[["amount"]])
    pdf["fraud_probability"] = model.predict_proba(pdf)[:,1]
    print(pdf[pdf["fraud_probability"] > 0.8])

features.writeStream.foreachBatch(process_batch).start().awaitTermination()