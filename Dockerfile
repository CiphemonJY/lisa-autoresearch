# LISA+Offload - Docker Image
# One-command deployment

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlx \
    mlx-lm \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Create app directory
WORKDIR /app

# Copy LISA+Offload files
COPY lisa_offload.py .
COPY disk_offload.py .
COPY hardware_detection.py .
COPY mixed_precision.py .
COPY gradient_accumulation.py .
COPY selective_offload.py .
COPY api_server.py .

# Create directories
RUN mkdir -p /app/models /app/data /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "api_server.py"]