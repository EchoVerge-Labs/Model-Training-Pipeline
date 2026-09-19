# NGC PyTorch container for DGX Spark (ARM64 / GB10)
# Pin by tag — for production, pin by digest as well
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        rclone ffmpeg sox libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — change requirements.txt to bust)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install benchmark harness
RUN pip install --no-cache-dir \
    "slsb @ git+https://github.com/EchoVerge-Labs/SLSB-benchmark.git@v0.1.0"

# Project code (changes most often → last layer)
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]
