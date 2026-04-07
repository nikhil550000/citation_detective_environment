FROM python:3.11-slim-bookworm

WORKDIR /app

# Install git (needed for pip install from GitHub)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy all environment code
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=11.0" \
    "openai>=1.0.0" \
    "requests>=2.28.0"

# Install openenv from GitHub
RUN pip install --no-cache-dir "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git" || \
    pip install --no-cache-dir git+https://github.com/meta-pytorch/OpenEnv.git || \
    echo "openenv install failed, continuing"

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
