# Optimized Dockerfile with Better Layer Caching
# Base Image with PyTorch and CUDA
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# LAYER 1: System Dependencies (Changes: Rarely)
# ============================================================================
RUN apt-get clean && \
    (apt-get update || apt-get update) && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================================
# LAYER 2: Python Dependencies from requirements.txt (Changes: Occasionally)
# ============================================================================
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir nats_bench>=1.5

# ============================================================================
# LAYER 3: External Dependencies - xautodl (Changes: Rarely)
# ============================================================================
# Install from git - this is stable and rarely changes
RUN pip install --no-cache-dir git+https://github.com/D-X-Y/AutoDL-Projects.git

# ============================================================================
# LAYER 4: alpha-beta-CROWN (Changes: Rarely)
# ============================================================================
# Copy alpha-beta-CROWN directory and install
# This is large but stable - changes infrequently
COPY alpha-beta-CROWN/ ./alpha-beta-CROWN/
# CRITICAL FIX: auto_LiRPA installs wrong onnx2pytorch, must reinstall correct version
RUN cd alpha-beta-CROWN && pip install -e auto_LiRPA && \
    pip uninstall -y onnx2pytorch && \
    pip install --no-cache-dir git+https://github.com/Verified-Intelligence/onnx2pytorch@fe7281b9b6c8c28f61e72b8f3b0e3181067c7399

# ============================================================================
# LAYER 5: Static Data - NATS-Bench (Changes: Never)
# ============================================================================
# This is the largest file (~1GB) and never changes
# Copy it after dependencies to maximize cache hits
RUN mkdir -p /root/.torch
COPY data/NATS-tss-v1_0-3ffb9-simple /root/.torch/NATS-tss-v1_0-3ffb9-simple

# ============================================================================
# LAYER 6: Configuration Files (Changes: Occasionally)
# ============================================================================
# Config files change less frequently than source code
COPY config/ ./config/

# ============================================================================
# LAYER 7: Source Code (Changes: Frequently)
# ============================================================================
# Copy source code - this changes most frequently
# Placed after all dependencies and config to minimize rebuild impact
COPY src/ ./src/

# ============================================================================
# LAYER 8: Scripts (Changes: Very Frequently)
# ============================================================================
# Scripts are modified most often during development
# Placed last so changes only rebuild this final layer
COPY scripts/ ./scripts/
COPY deployment.md .

# ============================================================================
# Environment and Entrypoint
# ============================================================================
ENV PYTHONPATH="/app"

CMD ["bash", "scripts/run_full_pipeline.sh"]
