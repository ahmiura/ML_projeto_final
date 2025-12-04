import sys
import os
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from src.etl.processor import extract_data, transform_data, load_data
from src.models.trainer import train_sentiment_model
from src.monitoring.service import PredictionMonitoring

default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- Configurações lidas do ambiente (via .env) ---
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")
db_host = "postgres_app"
db_name = os.getenv("POSTGRES_DB_APP")

DB_CONN = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"

def check_model_performance():
    """
    Verifica se o modelo precisa fazer retraining comparando 
    a distribuição de sentimentos recente vs histórica
    """
    print("🔍 Checando performance do modelo...")
    
    monitoring = PredictionMonitoring(DB_CONN)
    drift_df = monitoring.detect_drift()
    
    if drift_df.empty:
        print("⚠️ Nenhum dado de drift para analisar. Pulando retraining.")
        return 'task_skip_retrain'
    
    # Verifica se há variação significativa (> 20%)
    variacao_maxima = drift_df['variacao_percentual'].abs().max()
    
    print(f"Variação máxima detectada: {variacao_maxima:.2f}%")
    
    if variacao_maxima > 20:
        print("🔴 DRIFT DETECTADO! Iniciando retraining...")
        return 'task_retrain'
    else:
        print("✅ Modelo com performance OK. Não há drift significativo.")
        return 'task_skip_retrain'


def verify_low_confidence_predictions():
    """
    Conta predições com baixa confiança (< 60%).
    Se > 5% do total, sinaliza para análise.
    """
    print("📊 Analisando confiança das predições...")
    
    monitoring = PredictionMonitoring(DB_CONN)
    low_conf = monitoring.get_low_confidence_predictions(threshold=0.6, limit=1000)
    all_preds = monitoring.get_metrics_by_period(days=1)
    
    total_predicoes = all_preds['total_predicoes'].sum()
    percentual_baixa_conf = (len(low_conf) / total_predicoes * 100) if total_predicoes > 0 else 0
    
    print(f"Predições com baixa confiança: {percentual_baixa_conf:.2f}%")
    
    if percentual_baixa_conf > 5:
        print("⚠️ Taxa alta de baixa confiança. Recomenda-se revisão manual.")
        return 'task_alert_team'
    
    return 'task_continue'


def notify_team(context=None):
    """
    Envia notificação para o time (pode integrar com Slack, email, etc)
    """
    print("📧 Notificando o time sobre anomalias detectadas...")
    # TODO: Integrar com Slack/Email
    pass


with DAG(
    dag_id="sentiment_model_monitoring",
    default_args=default_args,
    description="Monitora performance do modelo e faz retraining se necessário",
    schedule_interval="0 2 * * *",  # Executa diariamente às 2 da manhã
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ml", "monitoring"]
) as dag:

    # Task 1: Checa performance e decide se faz retraining
    task_check = BranchPythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
        doc="Detecta data drift comparando distribuições recente vs histórica"
    )

    # Task 2: Treina novo modelo se necessário
    task_retrain = PythonOperator(
        task_id="task_retrain",
        python_callable=train_sentiment_model,
        op_kwargs={"db_connection_str": DB_CONN},
        doc="Reaplica o pipeline de treinamento com novos dados"
    )

    # Task 3: Pula retraining se não houver drift
    task_skip = PythonOperator(
        task_id="task_skip_retrain",
        python_callable=lambda: print("✅ Modelo mantido. Nenhum retraining necessário.")
    )

    # Task 4: Verifica confiança das predições
    task_verify_conf = PythonOperator(
        task_id="verify_confidence",
        python_callable=verify_low_confidence_predictions
    )

    # Task 5: Alerta o time se necessário
    task_alert = PythonOperator(
        task_id="task_alert_team",
        python_callable=notify_team,
        trigger_rule="none_failed_min_one_success"
    )

    task_continue = PythonOperator(
        task_id="task_continue",
        python_callable=lambda: print("✅ Nível de confiança OK")
    )

    # Define a sequência de execução
    task_check >> [task_retrain, task_skip]
    task_skip >> task_verify_conf >> [task_alert, task_continue]
