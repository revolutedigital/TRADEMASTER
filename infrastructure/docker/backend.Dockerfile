FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup appuser

# Install Python dependencies (including ML extras)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,ml]"

# Copy application code
COPY . .

# Create directories for ML artifacts with correct ownership
RUN mkdir -p ml_artifacts/models ml_artifacts/scalers && \
    chown -R appuser:appgroup ml_artifacts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
