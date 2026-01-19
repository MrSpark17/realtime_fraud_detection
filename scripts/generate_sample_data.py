import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "historical_transactions.csv")

def generate_sample_data(n_samples=1000, fraud_ratio=0.05):
    """
    Generate sample transaction data for model training.
    
    Args:
        n_samples: Total number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (0.0-1.0)
    
    Returns:
        DataFrame with transaction data
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_data = {
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_normal)],
        "user_id": np.random.randint(1000, 5000, n_normal),
        "amount": np.random.uniform(10, 500, n_normal),
        "merchant_id": np.random.randint(100, 999, n_normal),
        "location": np.random.choice(["US", "IN", "UK", "SG"], n_normal),
        "timestamp": np.random.randint(1600000000, 1700000000, n_normal),
        "is_high_value": (np.random.uniform(0, 1, n_normal) > 0.95).astype(int),
        "is_fraud": np.zeros(n_normal, dtype=int)
    }
    
    # Fraudulent transactions (higher amounts, unusual patterns)
    fraud_data = {
        "transaction_id": [f"TXN_{i+n_normal:06d}" for i in range(n_fraud)],
        "user_id": np.random.randint(1000, 5000, n_fraud),
        "amount": np.random.uniform(2000, 5000, n_fraud),  # Typically high-value
        "merchant_id": np.random.randint(100, 999, n_fraud),
        "location": np.random.choice(["US", "IN", "UK", "SG"], n_fraud),
        "timestamp": np.random.randint(1600000000, 1700000000, n_fraud),
        "is_high_value": np.ones(n_fraud, dtype=int),  # Usually high-value
        "is_fraud": np.ones(n_fraud, dtype=int)
    }
    
    # Combine and shuffle
    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating sample transaction data...")
    df = generate_sample_data(n_samples=1000, fraud_ratio=0.05)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(DATA_PATH, index=False)
    print(f"✓ Sample data saved to {DATA_PATH}")
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"  - Columns: {', '.join(df.columns.tolist())}")
