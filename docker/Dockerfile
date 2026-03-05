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

## Copy Python dependencies first (Docker layer caching)
COPY requirements.txt /app/requirements.txt

## Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

## Copy full project
COPY . /app

## Expose FastAPI + Streamlit ports
EXPOSE 8000
EXPOSE 8501

## Default command: run API (FastAPI via main.py)
CMD ["python", "main.py", "--run-api", "--host", "0.0.0.0", "--port", "8000"]