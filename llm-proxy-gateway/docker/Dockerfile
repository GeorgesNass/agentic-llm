## ============================================================
## LLM Proxy Gateway - Dockerfile
## ============================================================

## Use a lightweight Python base image
FROM python:3.11-slim

## Disable Python bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1

## Force unbuffered output for logs
ENV PYTHONUNBUFFERED=1

## Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

## Set working directory inside the container
WORKDIR /app

## Copy Python dependencies file first (Docker cache optimization)
COPY requirements.txt /app/requirements.txt

## Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

## Copy the full project source code
COPY . /app

## Expose FastAPI port
EXPOSE 8000

## Default command: run API service
CMD ["python", "main.py", "--run-api", "--host", "0.0.0.0", "--port", "8000"]