import sys
import os
# Adiciona a pasta raiz (/opt/airflow) ao caminho de busca do Python
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.models.trainer import train_sentiment_model
from src.etl.processor import extract_data, transform_data, load_data

default_args = {
    "owner": "mlops_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- Configurações lidas do ambiente (via .env) ---

# Monta a string de conexão a partir de variáveis de ambiente
db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB_APP", "bacen")
db_host = "postgres_app" # Nome do serviço no docker-compose
DB_CONN = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"

# Caminhos para o pipeline
RAW_CSV = os.getenv("RAW_CSV_PATH", "/opt/airflow/data/olist_order_reviews_dataset.csv")
PATH_RAW = os.getenv("TEMP_RAW_PATH", "/tmp/reviews_raw.parquet")
PATH_REFINED = os.getenv("TEMP_REFINED_PATH", "/tmp/reviews_clean.parquet")


with DAG(
    dag_id="sentiment_initial_etl_and_training",
    default_args=default_args,
    description="Pipeline de ETL e treinamento inicial do modelo de sentimento com dados de reviews da Olist.",
    # Este pipeline deve ser executado manualmente ou apenas uma vez, não diariamente.
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # 1. Extração
    task_extract = PythonOperator(
        task_id="extract",
        python_callable=extract_data,
        op_kwargs={"input_path": RAW_CSV, "output_path": PATH_RAW}
    )

    # 2. Transformação (Lê o output da extração)
    task_transform = PythonOperator(
        task_id="transform",
        python_callable=transform_data,
        op_kwargs={"input_path": PATH_RAW, "output_path": PATH_REFINED}
    )

    # 3. Carga (Lê o output da transformação)
    task_load = PythonOperator(
        task_id="load",
        python_callable=load_data,
        op_kwargs={
            "input_path": PATH_REFINED, 
            "db_connection_str": DB_CONN
        }
    )

    # 4. Treinamento (Lê do Banco de Dados)
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_sentiment_model,
        op_kwargs={"db_connection_str": DB_CONN},
    )

    # Definição do Fluxo
    task_extract >> task_transform >> task_load >> task_train