FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy all environment code
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=11.0" \
    "google-generativeai>=0.8.0" \
    "openai>=1.0.0" \
    "requests>=2.28.0"

# Install openenv from GitHub (not on PyPI)
RUN pip install --no-cache-dir "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git" || \
    echo "openenv install failed, continuing"

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

# Use port 7860 (HF Spaces default)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
