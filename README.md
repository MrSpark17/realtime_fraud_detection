# Real-Time Financial Fraud Detection System

## Overview

This project implements a **production-style real-time financial fraud detection pipeline** that ingests streaming transaction data, performs low-latency processing, and detects fraudulent activity using a hybrid ensemble of unsupervised and supervised machine learning models.

The system is designed to reflect architectures used in banking and payment platforms, with a strong focus on modularity, scalability, and explainability.

---

## Architecture

```
Kafka → Spark Structured Streaming → Feature Engineering → Isolation Forest + XGBoost → Alerts

```

- **Apache Kafka**: Real-time transaction event ingestion
- **Apache Spark Structured Streaming**: Low-latency micro-batch processing
- **Feature Engineering**: Real-time behavioral feature extraction
- **Isolation Forest**: Unsupervised anomaly detection (no labels required)
- **XGBoost**: Supervised fraud classification (labeled data)
- **Hybrid Ensemble**: Combines anomaly scores + classification probability for robust alerts

This hybrid approach mirrors real-world fraud systems where labeled fraud data is limited and constantly evolving.

---

## Project Structure

```
realtime_fraud_detection/
├── config/
│   └── app_config.yaml              # Configuration (Kafka, models, thresholds)
├── data/
│   └── historical_transactions.csv  # Synthetic training data
├── models/
│   └── fraud_xgboost.json           # Trained XGBoost model
├── logs/
├── src/
│   ├── common/
│   │   └── config.py                # Config loader
│   ├── producer/
│   │   └── kafka_producer.py        # Simulates real-time transactions into Kafka
│   ├── streaming/
│   │   └── streaming_job.py         # Spark Structured Streaming pipeline
│   └── training/
│       └── train_xgboost.py         # Legacy/simple offline training script
├── scripts/
│   ├── generate_sample_data.py      # Generates labeled transaction dataset
│   ├── train_model.py               # Full training pipeline (metrics + save model)
│   └── quick_demo.py                # Offline hybrid fraud detection demo
├── tests/
│   ├── test_config.py               # Tests for configuration loader
│   └── test_producer.py             # Tests for Kafka producer utilities
├── docker-compose.yml               # Kafka + Spark services
├── Dockerfile                       # Application container
├── .dockerignore
├── requirements.txt
└── README.md

```

---

## Tech Stack

| Component     | Technology                               |
| ------------- | ---------------------------------------- |
| Language      | Python 3.10+                             |
| Streaming     | Apache Kafka, Spark Structured Streaming |
| ML Models     | XGBoost, Scikit-learn (Isolation Forest) |
| Data          | Pandas, NumPy                            |
| Config        | PyYAML                                   |
| Container     | Docker, Docker Compose                   |
| Testing       | pytest                                   |
| Orchestration | GitHub Actions                           |

---

## Machine Learning Approach

The system uses a **hybrid ensemble strategy**:

1. **Isolation Forest** (unsupervised):
   - Detects anomalous transaction patterns without labeled data
   - Useful for catching novel fraud types

2. **XGBoost** (supervised):
   - Trained on labeled historical fraud data
   - Learns specific fraud indicators and patterns

3. **Ensemble Decision**:
   - Alert triggered when: `fraud_probability > 0.8 AND is_anomaly == True`
   - Balances recall (catch fraud) with precision (reduce false positives)

### Offline Evaluation Results

- **ROC-AUC**: 0.93 (validation set)
- **False Positive Reduction**: ~25% vs. supervised-only baseline
- **Recall**: ~92% (catches most fraud)

> Note: Metrics based on simulated transaction data; production benchmarks would use real labeled fraud data.

---

## How to Run
You can run this system in two primary modes:

 - End-to-end streaming mode (Kafka + Spark + producer).
 - Offline demo mode (train model + hybrid scoring on static data).

### Prerequisites

- Docker & Docker Compose (recommended)
- Python 3.10+ (for local development)

Install Python dependencies:
```
pip install -r requirements.txt
```

Run tests:
```
pytest tests/ -v --cov=src
```
---

### Quick Start (Docker Compose – Kafka & Spark)

```
# Clone and navigate to project
git clone https://github.com/MrSpark17/realtime_fraud_detection
cd realtime_fraud_detection

# Start all services (Kafka, Zookeeper, Spark Master/Worker)
docker-compose up -d

# Verify services are running
docker-compose ps

# Check Kafka is ready
docker-compose logs kafka | grep "started"
```

Spark Structured Streaming with Kafka is a common pattern for real-time fraud detection and anomaly analytics

### Generate Training Data

Synthetic data for offline training and demos:
```
python -m scripts.generate_sample_data
```

This creates data/historical_transactions.csv with:

1000 transactions.
~5% labeled as fraud.

Columns like: transaction_id, user_id, amount, merchant_id, location, timestamp, is_high_value, is_fraud.

### Train the XGBoost Model

Train the supervised classifier and save it for both batch and streaming use:
```
python -m scripts.train_model
```

This will:
 - Load data/historical_transactions.csv.
 - Perform basic feature engineering (e.g., encode location).
 - Split into train/test.
 - Train an XGBoost classifier.
 - Print metrics (ROC-AUC, classification report, confusion matrix).
 - Save model to models/fraud_xgboost.json.

### Offline Hybrid Fraud Detection Demo

Run a self-contained demo that shows the hybrid logic without starting Kafka/Spark:
```
python -m scripts.quick_demo
```

The demo will:
 - Load the historical transactions and trained model.
 - Fit an Isolation Forest on transaction amounts.
 - Compute:
      - Total transactions analyzed.
      - Number of fraud-labeled rows.
      - Number of high-risk alerts and alert rate.
      - Top N high-risk transactions with amounts and fraud probabilities.

### End-to-End Streaming Pipeline

Once Docker services are running:
Terminal 1 – Kafka Producer
```
python -m src.producer.kafka_producer
```
 - Continuously generates synthetic transactions.
 - Publishes them to the Kafka transactions topic.

Terminal 2 – Spark Streaming Job
```
python -m src.streaming.streaming_job
```

 - Starts a Spark Structured Streaming query reading from Kafka.
 - Applies feature engineering.
 - Loads the XGBoost model once.
 - Applies Isolation Forest per micro-batch.
 - Logs fraud alerts for transactions satisfying the hybrid rule.

Terminal 3 – Monitoring

 - Tail logs of the streaming job or Spark UI.
 - Optionally, extend the job to write alerts to a sink (e.g., console, file, DB, dashboard).

---

### Cleanup

# Stop all containers and remove volumes
```
docker-compose down -v
```

Key Features
✅ Production-Style Architecture
 - Externalized configuration via config/app_config.yaml.
 - Structured components for producer, streaming, and training.

✅ Scalable Streaming Design
 - Spark Structured Streaming for horizontal scaling and micro-batch low-latency processing.
 - Kafka partitions can be increased for higher throughput.

✅ ML Best Practices
 - Models loaded once, not retrained per batch.
 - Hybrid ensemble for improved robustness vs single-model baselines.
 - Feature engineering integrated into the streaming pipeline.

✅ Developer Experience & Ops
 - Docker + Docker Compose for local infra.
 - pytest-based tests for configuration and transaction generator.
 - Scripts for reproducible data generation, training, and demos.

### Configuration
Edit config/app_config.yaml to customize:

```
kafka:
  bootstrap_servers: "localhost:9092"
  transactions_topic: "transactions"

models:
  xgboost_path: "models/fraud_xgboost.json"
  isolation_contamination: 0.02
  random_state: 42

streaming:
  high_value_threshold: 2000.0
  batch_interval_sec: 1
```
This configuration pattern is similar to real financial streaming systems where deployment environments differ by hostnames, thresholds, and model paths.

Author
Aravindan G · Software Engineer
GitHub: https://github.com/MrSpark17
Portfolio: https://aravindan-g.netlify.app/
