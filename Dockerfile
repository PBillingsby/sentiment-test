FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded models into the container
RUN mkdir -p /models && \
COPY models /models

# Create output directory
RUN mkdir -p /outputs

# Copy source code
COPY src /src

ENV HF_HOME=/model \
TRANSFORMERS_OFFLINE=1

# Set entrypoint
ENTRYPOINT ["python", "/src/run_inference.py"]
