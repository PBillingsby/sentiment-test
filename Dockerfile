FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add models
RUN mkdir /models
ADD https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english /models

# Create output directory
RUN mkdir -p /outputs

# Copy source code
COPY src /src

ENV HF_HOME=/model \
TRANSFORMERS_OFFLINE=1

# Set entrypoint
ENTRYPOINT ["python", "/src/run_inference.py"]
