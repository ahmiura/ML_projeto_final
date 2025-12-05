import sys
import os
# Adiciona a pasta raiz do projeto ao path para que o Airflow encontre os módulos
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- Configurações lidas do ambiente (via .env) ---
db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_host = "postgres_app"
db_name = os.getenv("POSTGRES_DB_APP", "bacen")

DB_CONN = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"

# --- Constantes de Gatilho ---
DRIFT_THRESHOLD = 20.0  # Variação percentual máxima antes de acionar retreino
MIN_FEEDBACK_COUNT = 50 # Número mínimo de novos feedbacks para acionar retreino

def check_retraining_triggers(**kwargs):
    """
    Verifica os gatilhos para o retreinamento.
    1. Detecta Data Drift significativo.
    2. Verifica se há um número suficiente de novos feedbacks.
    Retorna a task_id a ser executada em seguida ('trigger_retraining' ou 'skip_retraining').
    """
    # A importação é feita dentro da função para ser compatível com o Airflow
    from src.monitoring.service import PredictionMonitoring
    from src.db.predictions import PredictionsRepo
    from sqlalchemy import create_engine

    logger = kwargs['ti'].log
    logger.info("--- Verificando Gatilhos para Retreinamento ---")
    
    monitoring = PredictionMonitoring(DB_CONN)
    
    # Gatilho 1: Data Drift
    logger.info("1. Verificando Data Drift...")
    drift_df = monitoring.detect_drift()
    has_drift = False
    if not drift_df.empty:
        max_variation = drift_df['variacao_percentual'].abs().max()
        logger.info(f"Variação máxima de distribuição detectada: {max_variation:.2f}%")
        if max_variation > DRIFT_THRESHOLD:
            logger.warning(f"🔴 DRIFT DETECTADO! Variação ({max_variation:.2f}%) acima do limite ({DRIFT_THRESHOLD}%).")
            has_drift = True
        else:
            logger.info("✅ Nível de drift aceitável.")
    else:
        logger.info("Nenhum dado de drift para analisar.")

    # Gatilho 2: Novos Feedbacks
    logger.info("2. Verificando quantidade de novos feedbacks...")
    engine = create_engine(DB_CONN)
    repo = PredictionsRepo(engine)
    feedback_data = repo.get_feedback_data()
    feedback_count = len(feedback_data)
    logger.info(f"Encontrados {feedback_count} registros com feedback humano.")
    
    has_enough_feedback = False
    if feedback_count >= MIN_FEEDBACK_COUNT:
        logger.warning(f"🟢 FEEDBACK SUFICIENTE! Quantidade ({feedback_count}) atinge o limite ({MIN_FEEDBACK_COUNT}).")
        has_enough_feedback = True
    else:
        logger.info("✅ Quantidade de feedback abaixo do limite para retreino.")

    # Decisão Final
    if has_drift or has_enough_feedback:
        logger.info("DECISÃO: Iniciar o pipeline de retreinamento.")
        return 'trigger_retraining'
    else:
        logger.info("DECISÃO: Pular o retreinamento desta vez.")
        return 'skip_retraining'

with DAG(
    dag_id="sentiment_model_lifecycle",
    default_args=default_args,
    description="DAG unificada para monitorar e retreinar o modelo de sentimento.",
    schedule_interval="0 2 * * *",  # Executa diariamente às 2 da manhã para monitoramento
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["mlops", "sentiment-analysis", "lifecycle"]
) as dag:

    # Task 1: Verifica os gatilhos (Drift e Novos Dados) e decide o próximo passo.
    check_triggers_task = BranchPythonOperator(
        task_id="check_retraining_triggers",
        python_callable=check_retraining_triggers,
        doc="Decide se o retreinamento deve ser acionado com base em drift ou novos feedbacks."
    )

    # Task 2 (Caminho A): Executa o pipeline de retreinamento.
    retrain_task = BashOperator(
        task_id="trigger_retraining",
        # Usar o caminho absoluto para o script para evitar erros de "file not found"
        bash_command="python -m src.etl.retraining_pipeline",
        cwd="/opt/airflow", # Define o diretório de trabalho para a raiz do projeto
        doc="Executa o script que combina dados originais e de feedback para treinar e registrar um novo modelo."
    )

    # Task 2 (Caminho B): Apenas registra que o retreinamento foi pulado.
    skip_task = PythonOperator(
        task_id="skip_retraining",
        python_callable=lambda: print("✅ Gatilhos não atingidos. Nenhum retreinamento necessário.")
    )

    # Define a sequência de execução
    check_triggers_task >> [retrain_task, skip_task]
