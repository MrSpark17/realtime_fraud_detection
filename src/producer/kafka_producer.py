import json
import time
import random
import uuid
import logging

from kafka import KafkaProducer
from src.common.config import get_config

cfg = get_config()
kafka_cfg = cfg["kafka"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("kafka_producer")

def generate_transaction():
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": random.randint(1000, 5000),
        "amount": round(random.uniform(1, 5000), 2),
        "merchant_id": random.randint(100, 999),
        "location": random.choice(["US", "IN", "UK", "SG"]),
        "timestamp": int(time.time())
    }

def main():
    producer = KafkaProducer(
        bootstrap_servers=kafka_cfg["bootstrap_servers"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    logger.info("Kafka producer started, sending to topic '%s'", kafka_cfg["transactions_topic"])

    while True:
        tx = generate_transaction()
        producer.send(kafka_cfg["transactions_topic"], tx)
        logger.info("Sent transaction_id=%s amount=%.2f", tx["transaction_id"], tx["amount"])
        time.sleep(0.2)

if __name__ == "__main__":
    main()
