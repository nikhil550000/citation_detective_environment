FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy all environment code
COPY . /app/

# Install Python dependencies directly (no uv needed)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=11.0" \
    "google-generativeai>=0.8.0" \
    "openai>=1.0.0" \
    "requests>=2.28.0"

# Install openenv from PyPI
RUN pip install --no-cache-dir "openenv-core[core]>=0.2.2" || \
    pip install --no-cache-dir "openenv[core]>=0.2.0" || \
    echo "Warning: openenv not available on PyPI, continuing without it"

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
