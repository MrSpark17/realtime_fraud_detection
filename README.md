# Real-Time Financial Fraud Detection System

## Overview
This project implements a real-time financial fraud detection pipeline that ingests streaming transaction data, performs low-latency processing, and detects fraudulent activity using a combination of unsupervised and supervised machine learning models.

The system is designed to reflect production-style fraud detection architectures used in banking and payment platforms, with a strong focus on modularity, scalability, and explainable design choices.

---

## Architecture
**Kafka → Spark Structured Streaming → Feature Engineering → Isolation Forest → XGBoost → Output Sink**

- **Apache Kafka** simulates real-time transaction ingestion
- **Apache Spark Structured Streaming** processes transaction events with low latency
- **Feature Engineering** derives real-time behavioral features
- **Isolation Forest** identifies anomalous transaction patterns
- **XGBoost** performs supervised fraud classification
- **Output Sink** represents downstream alerting, storage, or monitoring systems

---

## Project Structure
src/
├── producer/
│ └── kafka_producer.py # Simulates real-time transaction events using Kafka
├── streaming/
│ └── streaming_job.py # Spark Structured Streaming fraud detection pipeline
└── training/
└── train_xgboost.py # Offline training pipeline for XGBoost model


---

## Tech Stack
- **Programming Language:** Python  
- **Streaming & Processing:** Apache Kafka, Apache Spark (Structured Streaming)  
- **Machine Learning:** XGBoost, Isolation Forest (Scikit-learn)  
- **Data Processing:** Pandas, NumPy  

---

## Machine Learning Approach
The fraud detection logic combines both **unsupervised** and **supervised** learning techniques:

- **Isolation Forest** is used to identify anomalous transaction behavior in streaming data without requiring labels.
- **XGBoost** is trained on labeled transaction data to perform supervised fraud classification.
- An ensemble approach combines anomaly scores and classification outputs to improve recall while reducing false positives.

This hybrid strategy mirrors real-world fraud detection systems where labeled fraud data is limited and continuously evolving.

---

## Results (Offline Evaluation)
- Achieved **ROC-AUC ≈ 0.93** on validation data  
- Reduced false positives by **~25%** compared to a baseline supervised-only model  

> Note: Metrics are based on offline evaluation using simulated transaction data and are intended to demonstrate model effectiveness rather than production benchmarks.

## System Architecture

```mermaid
flowchart LR
    A[Transaction Events] --> B[Kafka Producer]
    B --> C[(Kafka Topic)]

    C --> D[Spark Structured Streaming]

    D --> E[Feature Engineering]

    E --> F[Isolation Forest<br/>(Anomaly Detection)]
    F --> G[XGBoost<br/>(Fraud Classification)]

    G --> H[Output Sink<br/>(Alerts / Storage)]

---

## How to Run

### Prerequisites
- Python 3.8+
- Apache Kafka
- Apache Spark

### Installation
```bash
pip install -r requirements.txt

Run Kafka Producer
python src/producer/kafka_producer.py

Run Spark Streaming Job
python src/streaming/streaming_job.py
