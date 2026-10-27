# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13-slim
FROM python:${PYTHON_VERSION}

# Metadata
LABEL maintainer="ML Team"
LABEL description="Lending Club Default Risk FastAPI Inference Service"

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disc
# - PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
# - PYTHONPATH: Ensures src/ can be imported cleanly across the app
# - PORT: Application server port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies
# - libgomp1: OpenMP runtime required by XGBoost and scikit-learn
# - curl: Required for container healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained model artifacts
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Container healthcheck querying FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default entrypoint starts the FastAPI uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
