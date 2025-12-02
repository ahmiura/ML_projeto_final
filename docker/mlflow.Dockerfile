FROM ghcr.io/mlflow/mlflow:v2.10.0

# Instalar dependências para compilar o psycopg2 e o próprio pacote
RUN pip install --no-cache-dir psycopg2-binary