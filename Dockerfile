FROM python:3.12-slim

WORKDIR /app

# Install Java (needed for Spark)
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless && rm -rf /var/lib/apt/lists/*

# Copy project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.producer.kafka_producer"]
