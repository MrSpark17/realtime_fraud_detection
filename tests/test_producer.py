import pytest
from src.producer.kafka_producer import generate_transaction

def test_generate_transaction():
    """Test that transaction generation works."""
    tx = generate_transaction()
    
    assert "transaction_id" in tx
    assert "user_id" in tx
    assert "amount" in tx
    assert "merchant_id" in tx
    assert "location" in tx
    assert "timestamp" in tx
    
    # Validate types and ranges
    assert isinstance(tx["user_id"], int)
    assert 1000 <= tx["user_id"] <= 5000
    assert isinstance(tx["amount"], float)
    assert 1 <= tx["amount"] <= 5000
    assert tx["location"] in ["US", "IN", "UK", "SG"]
