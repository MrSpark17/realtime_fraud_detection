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
│   └── historical_transactions.csv  # Training data
├── models/
│   └── fraud_xgboost.json          # Trained XGBoost model
├── logs/
├── src/
│   ├── common/
│   │   └── config.py               # Config loader
│   ├── producer/
│   │   └── kafka_producer.py       # Simulates real-time transactions
│   ├── streaming/
│   │   └── streaming_job.py        # Spark Structured Streaming pipeline
│   └── training/
│       └── train_xgboost.py        # Offline model training
├── tests/
│   ├── test_config.py
│   └── test_producer.py
├── docker-compose.yml               # Kafka + Spark services
├── Dockerfile                        # Application container
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Streaming | Apache Kafka, Spark Structured Streaming |
| ML Models | XGBoost, Scikit-learn (Isolation Forest) |
| Data Processing | Pandas, NumPy |
| Configuration | PyYAML |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| CI/CD | GitHub Actions |

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

### Prerequisites

- Docker & Docker Compose (recommended)
- Python 3.10+ (for local development)

### Quick Start (Docker Compose)

```bash
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

Once services are running, in separate terminals:

```bash
# Terminal 1: Start the Kafka producer (sends simulated transactions)
python -m src.producer.kafka_producer

# Terminal 2: Start the Spark streaming job (processes & detects fraud)
python -m src.streaming.streaming_job

# Terminal 3: Monitor output
# Watch fraud alerts in Terminal 2's logs
```

### Development Setup (Local Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v --cov=src

# For production model training (requires training data)
python -m src.training.train_xgboost
```

### Cleanup

```bash
# Stop all containers and remove volumes
docker-compose down -v
```

---

## Key Features

✅ **Production-Ready Architecture**
- Externalized configuration (YAML)
- Structured logging
- Error handling & retries
- Modular, testable code

✅ **Scalable Design**
- Spark Structured Streaming for horizontal scaling
- Kafka partitions for parallel processing
- Micro-batch processing for low latency

✅ **ML Best Practices**
- Models loaded once (not refit per batch)
- Hybrid ensemble for robust predictions
- Feature engineering integrated into pipeline

✅ **Deployment-Ready**
- Docker containerization
- Docker Compose orchestration
- GitHub Actions CI/CD

---

## Configuration

Edit `config/app_config.yaml` to customize:

```yaml
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

---

## Next Steps for Production Deployment

- [ ] Integrate with real Kafka brokers
- [ ] Connect to real fraud labels for model retraining
- [ ] Add monitoring & alerting (Prometheus, Grafana)
- [ ] Implement model versioning & A/B testing (MLflow)
- [ ] Add data validation & schema enforcement
- [ ] Implement feature store for consistency
- [ ] Deploy to Kubernetes for high availability

---

## Author

Aravindan G | Software Engineer | [GitHub](https://github.com/MrSpark17)