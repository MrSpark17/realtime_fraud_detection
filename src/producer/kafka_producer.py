from kafka import KafkaProducer
import json, time, random, uuid

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": random.randint(1000, 5000),
        "amount": round(random.uniform(1, 5000), 2),
        "merchant_id": random.randint(100, 999),
        "location": random.choice(["US", "IN", "UK", "SG"]),
        "timestamp": int(time.time())
    }

while True:
    producer.send("transactions", generate_transaction())
    time.sleep(0.2)