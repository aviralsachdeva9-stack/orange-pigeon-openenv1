FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the inference script with LLM API calls for Phase 2 validation
CMD ["python", "inference.py"]