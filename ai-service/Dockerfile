# Multi-stage build for Python AI service
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Development stage — uvicorn with hot reload (dev compose target).
# ---------------------------------------------------------------------------
FROM base AS development

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---------------------------------------------------------------------------
# Production stage — no reload; worker count override with AI_WORKERS.
# Runs behind the edge Nginx, hence --proxy-headers.
# ---------------------------------------------------------------------------
FROM base AS production

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    --workers ${AI_WORKERS:-2} \
    --proxy-headers --forwarded-allow-ips '*'
