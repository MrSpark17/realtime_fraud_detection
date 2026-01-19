import os
import pytest
from src.common.config import get_config

def test_config_loads():
    """Test that config loads without errors."""
    cfg = get_config()
    assert cfg is not None
    assert "kafka" in cfg
    assert "models" in cfg
    assert "streaming" in cfg

def test_kafka_config():
    """Test Kafka config values."""
    cfg = get_config()
    kafka_cfg = cfg["kafka"]
    assert kafka_cfg["bootstrap_servers"] == "localhost:9092"
    assert kafka_cfg["transactions_topic"] == "transactions"

def test_models_config():
    """Test models config values."""
    cfg = get_config()
    model_cfg = cfg["models"]
    assert "xgboost_path" in model_cfg
    assert float(model_cfg["isolation_contamination"]) > 0
