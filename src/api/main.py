import time
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
import os
from typing import Optional, AsyncGenerator, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, text

# Importa a mesma função de limpeza usada no treinamento para garantir consistência
from src.etl.processor import clean_text
from src.monitoring.service import PredictionMonitoring

# Variáveis globais para armazenar modelo e monitoramento em memória
model = None

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
vectorizer = None
monitoring = None
model_name_loaded = None
model_registry_version = None
model_algorithm_loaded = None

# Carrega variáveis de ambiente para conexão com o banco
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")
db_host = os.getenv("POSTGRES_HOST", "postgres_app") # Se nãoo encontrar o default é postgres_app

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Código de inicialização (executa quando a API sobe)
    global model, vectorizer, monitoring, model_name_loaded, model_registry_version, model_algorithm_loaded
    try:
        logger.info("Buscando modelo de PRODUÇÃO no MLflow...")
        model_name = "modelo_sentimento_bacen"
        stage = "Production"
        
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
        
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=[stage])
        
        if not versions:
            logger.warning("Nenhum modelo em produção encontrado.")
        else:
            run_id = versions[0].run_id
            model_registry_version = versions[0].version
            model_name_loaded = model_name
            # Extrai nome do algoritmo/estimador de forma robusta
            def _extract_algorithm_name(m):
                try:
                    # sklearn Pipeline
                    if hasattr(m, "steps"):
                        return m.steps[-1][1].__class__.__name__
                    # sklearn Pipeline (named_steps)
                    if hasattr(m, "named_steps"):
                        return list(m.named_steps.values())[-1].__class__.__name__
                    # MLflow pyfunc ou estimator direto
                    return m.__class__.__name__
                except Exception:
                    return None

            model_algorithm_loaded = _extract_algorithm_name(model)
            logger.info(f"Modelo de Produção encontrado no Run ID: {run_id} (version={model_registry_version}, algo={model_algorithm_loaded})")
            vectorizer = mlflow.sklearn.load_model(f"runs:/{run_id}/vectorizer")
            
            # Inicializa o monitor de predições
            DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/bacen"
            monitoring = PredictionMonitoring(DATABASE_URL)
            
            logger.info("Modelo, Vetorizador e Monitoramento carregados com sucesso!")
            
    except Exception as e:
        logger.critical(f"Erro crítico ao carregar modelo: {e}", exc_info=True)
        raise RuntimeError(f"Não foi possível carregar o modelo: {e}")
    
    yield
    
    # Código de finalização (executa quando a API desliga)
    logger.info("API desligada. Limpando recursos se necessário.")

app = FastAPI(title="API de Análise de Sentimento de Clientes", lifespan=lifespan)

# Configurar MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

# --- MELHORIA: Criar o engine uma única vez ---
# Carregar a string de conexão de variáveis de ambiente para segurança
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/bacen"
# O engine é pesado, deve ser criado apenas uma vez quando a aplicação sobe.
# O pool de conexões será gerenciado automaticamente.
db_engine = create_engine(DATABASE_URL)

class CustomerMessageRequest(BaseModel):
    message: str

def log_prediction(
    texto: str, 
    predicao: str, 
    probabilidade: float, 
    tempo_ms: float
) -> None:
    """Função de log de predições na tabela logs_predicoes"""
    global model_name_loaded, model_registry_version, model_algorithm_loaded
    modelo_info = None
    if model_name_loaded and model_registry_version:
        modelo_info = f"{model_name_loaded}:{model_registry_version}"
        if model_algorithm_loaded:
            modelo_info = f"{modelo_info}|{model_algorithm_loaded}"
    df = pd.DataFrame([{
        "data": datetime.now(),
        "texto_input": texto,
        "classificacao": predicao,
        "probabilidade": probabilidade,
        "tempo_inferencia_ms": tempo_ms,
        "modelo_version": modelo_info
    }])
    df.to_sql("logs_predicoes", db_engine, if_exists="append", index=False)


@app.post("/predict")
def predict_sentiment(
    request: CustomerMessageRequest, 
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Endpoint principal para predição de sentimento
    
    Args:
        request: JSON com campo 'message' contendo o texto do cliente
        background_tasks: Para registrar log assincronamente
    
    Returns:
        JSON com sentimento, probabilidade, ação sugerida e tempo de inferência
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    inicio = time.time()  # Marca início da inferência
    
    # Pre-processamento
    text_clean = clean_text(request.message)
    vectorized = vectorizer.transform([text_clean])
    prediction = model.predict(vectorized)[0]
    proba_insatisfeito = model.predict_proba(vectorized)[0][1]
    
    tempo_ms = (time.time() - inicio) * 1000  # Converte para ms
    
    sentiment = "INSATISFEITO" if prediction == 1 else "SATISFEITO"
    
    # Registra log assincronamente para não bloquear a resposta
    background_tasks.add_task(
        log_prediction, 
        request.message, 
        sentiment, 
        float(proba_insatisfeito),
        tempo_ms
    )

    return {
        "sentimento": sentiment,
        "probabilidade_insatisfeito": float(proba_insatisfeito),
        "acao_sugerida": "TRANSBORDO_HUMANO" if prediction == 1 else "CONTINUAR_CHATBOT",
        "tempo_ms": round(tempo_ms, 2)
    }


@app.post("/feedback/{prediction_id}")
def submit_feedback(
    prediction_id: int, 
    feedback: str, 
    corrected_class: Optional[str] = None
) -> Dict[str, Any]:
    """
    Registra feedback humano para análise posterior e possível retraining
    
    Args:
        prediction_id: ID da predição na tabela logs_predicoes
        feedback: Tipo de feedback (ex: "CORRETO" ou "INCORRETO")
        corrected_class: Classificação corrigida se necessário (ex: "SATISFEITO" ou "INSATISFEITO")
    
    Exemplo:
        POST /feedback/123?feedback=INCORRETO&corrected_class=SATISFEITO
    """
    try:
        if monitoring is None:
            raise HTTPException(status_code=503, detail="Monitoramento não disponível")
        
        monitoring.log_feedback(prediction_id, feedback, corrected_class)
        
        return {
            "status": "success",
            "message": "Feedback registrado com sucesso",
            "prediction_id": prediction_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao registrar feedback: {str(e)}")


@app.get("/metrics")
def get_metrics(days: int = 7) -> Dict[str, Any]:
    """
    Retorna métricas agregadas e detecção de data drift
    
    Args:
        days: Número de dias para análise (default: 7)
    
    Returns:
        JSON com métricas por dia e detecção de drift
    """
    try:
        if monitoring is None:
            raise HTTPException(status_code=503, detail="Monitoramento não disponível")
        
        metrics_df = monitoring.get_metrics_by_period(days=days)
        drift_df = monitoring.detect_drift()
        
        return {
            "periodo_dias": days,
            "metricas": metrics_df.to_dict('records'),
            "deteccao_drift": drift_df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar métricas: {str(e)}")


@app.get("/low-confidence")
def get_low_confidence_predictions(
    threshold: float = 0.6, 
    limit: int = 50
) -> Dict[str, Any]:
    """
    Retorna predições com baixa confiança para revisão manual
    
    Args:
        threshold: Limite de confiança (default: 0.6 = 60%)
        limit: Número máximo de registros (default: 50)
    
    Exemplo:
        GET /low-confidence?threshold=0.7&limit=100
    """
    try:
        if monitoring is None:
            raise HTTPException(status_code=503, detail="Monitoramento não disponível")
        
        low_conf = monitoring.get_low_confidence_predictions(threshold, limit)
        
        return {
            "total": len(low_conf),
            "threshold": threshold,
            "predicoes": low_conf.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar predições: {str(e)}")