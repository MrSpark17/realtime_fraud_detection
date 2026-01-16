# Real-Time Financial Fraud Detection System

## Overview
This project implements a real-time financial fraud detection pipeline that ingests streaming transaction data, performs low-latency processing, and detects fraudulent activity using a combination of unsupervised and supervised machine learning models.

The system is designed to reflect production-style fraud detection architectures used in banking and payment platforms, with a strong focus on modularity, scalability, and explainable design choices.

---

## Architecture
Kafka → Spark Structured Streaming → Feature Engineering → Isolation Forest → XGBoost → Output Sink

- **Apache Kafka** simulates real-time transaction ingestion  
- **Apache Spark Structured Streaming** processes transaction events with low latency  
- **Feature Engineering** derives real-time behavioral features  
- **Isolation Forest** identifies anomalous transaction patterns  
- **XGBoost** performs supervised fraud classification  
- **Output Sink** represents downstream alerting, storage, or monitoring systems  

---

## System Architecture Diagram

```mermaid
flowchart LR
    A[Transaction Events] --> B[Kafka Producer]
    B --> C[(Kafka Topic)]

    C --> D[Spark Structured Streaming]
    D --> E[Feature Engineering]

    E --> F[Isolation Forest (Anomaly Detection)]
    F --> G[XGBoost (Fraud Classification)]

    G --> H[Output Sink (Alerts / Storage)]
