import logging
import os
import mlflow
import pandas as pd
from sqlalchemy import create_engine, text

from src.db.predictions import PredictionsRepo
from src.etl.processor import clean_text
from src.models.trainer import run_training_flow

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configurações ---

# Conexão com o Banco de Dados (usando variáveis de ambiente)
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_app")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}/bacen"

# Conexão com o MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Nome do modelo no Model Registry
MODEL_NAME = "modelo_sentimento_bacen"


def run_retraining_pipeline():
    """
    Executa o pipeline de retreinamento completo:
    1. Extrai dados originais (CSV) e de feedback (DB).
    2. Transforma e combina os dados.
    3. Treina um novo modelo e o registra no MLflow.
    """
    logger.info("--- Iniciando Pipeline de Retreinamento ---")

    try:
        # --- SETUP ---
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        db_engine = create_engine(DATABASE_URL)
        predictions_repo = PredictionsRepo(db_engine)

        # --- 1. EXTRACT ---
        # Extrair dados originais da Feature Store (tabela reviews_features)
        logger.info("Lendo dados originais da tabela 'reviews_features'...")
        with db_engine.connect() as conn:
            # Estes dados já estão limpos e com target definido
            original_df = pd.read_sql(text("SELECT texto_limpo, target FROM reviews_features"), conn)
        logger.info(f"Encontrados {len(original_df)} registros na Feature Store.")

        # Extrair dados de feedback
        logger.info("Buscando dados com feedback do banco de dados...")
        feedback_data = predictions_repo.get_feedback_data()
        
        if not feedback_data:
            logger.warning("Nenhum dado novo com feedback encontrado. O retreinamento não adicionará novos dados.")
            # Criamos um DataFrame vazio para o processamento a seguir não falhar
            feedback_df = pd.DataFrame(columns=['texto_limpo', 'target'])
        else:
            # Processa os dados de feedback, que vêm em formato "bruto"
            feedback_df_raw = pd.DataFrame(feedback_data, columns=['texto_input', 'classificacao'])
            feedback_df = pd.DataFrame() # Cria um novo DF para os dados processados
            feedback_df['texto_limpo'] = feedback_df_raw['texto_input'].apply(clean_text)
            feedback_df['target'] = feedback_df_raw['classificacao'].map({'SATISFEITO': 0, 'INSATISFEITO': 1})
            logger.info(f"Encontrados {len(feedback_df_raw)} registros com feedback para retreinamento.")

        # --- 2. TRANSFORM ---
        logger.info("Combinando e processando datasets...")
        # Concatena os dados originais (já limpos) com os de feedback (agora limpos)
        combined_df = pd.concat([original_df, feedback_df], ignore_index=True)        
        
        # Remover nulos que podem surgir de textos vazios ou mapeamentos falhos
        combined_df.dropna(subset=['texto_limpo', 'target'], inplace=True)
        combined_df = combined_df[combined_df['texto_limpo'] != '']
        combined_df['target'] = combined_df['target'].astype(int)

        X = combined_df['texto_limpo']
        y = combined_df['target']
        
        logger.info(f"Dataset final para treinamento com {len(combined_df)} registros.")

        # --- 3. TRAIN (usando o fluxo de treinamento centralizado) ---
        logger.info("Delegando para o fluxo de treinamento centralizado...")
        
        # Parâmetros sobre a origem dos dados para logar no MLflow
        data_params = {
            "num_feature_store_samples": len(original_df),
            "num_feedback_samples": len(feedback_df[feedback_df['target'].notna()]), # Conta apenas os que foram processados
            "num_total_samples": len(combined_df)
        }

        # Chama a função centralizada, que cuidará de todo o ciclo de vida do treinamento
        result_message = run_training_flow(combined_df, experiment_name="sentiment_retraining", data_params=data_params)
        logger.info(result_message)
        logger.info("--- Pipeline de Retreinamento Concluído com Sucesso ---")

    except Exception as e:
        logger.error(f"Falha crítica no pipeline de retreinamento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_retraining_pipeline()
