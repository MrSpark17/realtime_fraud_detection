import pytest
from src.producer.kafka_producer import generate_transaction

def test_generate_transaction():
    """Test that transaction generation works correctly."""
    tx = generate_transaction()
    
    # Check all fields exist
    assert "transaction_id" in tx
    assert "user_id" in tx
    assert "amount" in tx
    assert "merchant_id" in tx
    assert "location" in tx
    assert "timestamp" in tx
    
    # Validate types
    assert isinstance(tx["user_id"], int)
    assert isinstance(tx["amount"], float)
    assert isinstance(tx["location"], str)
    assert isinstance(tx["timestamp"], int)
    
    # Validate ranges
    assert 1000 <= tx["user_id"] <= 5000
    assert 1 <= tx["amount"] <= 5000
    assert tx["location"] in ["US", "IN", "UK", "SG"]
    assert tx["timestamp"] > 0

def test_generate_transaction_uniqueness():
    """Test that generated transactions have unique IDs."""
    tx1 = generate_transaction()
    tx2 = generate_transaction()
    assert tx1["transaction_id"] != tx2["transaction_id"]

def test_generate_transaction_multiple():
    """Test generating multiple transactions."""
    transactions = [generate_transaction() for _ in range(100)]
    assert len(transactions) == 100
    
    # Check all have unique IDs
    ids = [tx["transaction_id"] for tx in transactions]
    assert len(set(ids)) == 100
