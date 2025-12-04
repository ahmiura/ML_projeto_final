# Use uma imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho no container
WORKDIR /app

# --- Instala compiladores C++ ---
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libre2-dev \
    && rm -rf /var/lib/apt/lists/*

# -- Instala pybind11 ---
RUN pip install --no-cache-dir pybind11

# --- Instala dependências do requirements.txt ---
COPY requirements_frontend.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
