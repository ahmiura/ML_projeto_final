import sys
import os
# Adiciona a pasta raiz (/opt/airflow) ao caminho de busca do Python
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.etl.processor import extract_data, transform_data, load_data

default_args = {
    "owner": "data_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# @postgres_app = Nome do serviço no docker-compose
# /bacen = Nome do banco de dados definido no docker-compose
DB_CONN = "postgresql+psycopg2://airflow:airflow123@postgres_app/bacen"
RAW_CSV = "/opt/airflow/data/olist_order_reviews_dataset.csv"

# Caminhos temporários para troca de dados entre tasks
PATH_RAW = "/tmp/reviews_raw.parquet"
PATH_REFINED = "/tmp/reviews_clean.parquet"

# Função wrapper para chamar o trainer com o argumento correto
def task_train_callable(db_connection_str):
    # importa apenas na execução da task
    from src.models.trainer import train_sentiment_model
    return train_sentiment_model(db_connection_str)


with DAG(
    dag_id="sentiment_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
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
        python_callable=task_train_callable,
        op_kwargs={"db_connection_str": DB_CONN}
    )

    # Definição do Fluxo
    task_extract >> task_transform >> task_load >> task_train