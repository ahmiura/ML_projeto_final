import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

class PredictionMonitoring:
    def __init__(self, db_connection_str):
        self.engine = create_engine(db_connection_str)
    
    def get_metrics_by_period(self, days=7):
        """Calcula métricas dos últimos N dias"""
        query = text("""
        SELECT 
            DATE_TRUNC('day', data) as dia,
            COUNT(*) as total_predicoes,
            SUM(CASE WHEN classificacao = 'INSATISFEITO' THEN 1 ELSE 0 END) as insatisfeitos,
            AVG(probabilidade) as prob_media,
            SUM(CASE WHEN probabilidade < :threshold THEN 1 ELSE 0 END) as baixa_confianca
        FROM logs_predicoes
        WHERE data >= NOW() - INTERVAL :days_literal
        GROUP BY dia
        ORDER BY dia DESC
        """)
        # params: safer way to pass the interval as literal string
        params = {"threshold": 0.6, "days_literal": f"{int(days)} days"}
        return pd.read_sql(query, self.engine, params=params)
    
    def detect_drift(self, days=7):
        """Detecta mudanças no padrão de sentimentos (Data Drift)"""
        query = text("""
        WITH recent AS (
            SELECT classificacao, COUNT(*) as count
            FROM logs_predicoes
            WHERE data >= NOW() - INTERVAL :days_literal
            GROUP BY classificacao
        ),
        historical AS (
            SELECT classificacao, COUNT(*) as count
            FROM logs_predicoes
            WHERE data >= NOW() - INTERVAL '30 days'
              AND data < NOW() - INTERVAL :days_literal
            GROUP BY classificacao
        )
        SELECT 
            r.classificacao,
            r.count as qtde_recente,
            COALESCE(h.count, 0) as qtde_historica,
            -- evita erro de tipo no ROUND convertendo para numeric e tratando divisão por zero
            ROUND(
              COALESCE(
                ((r.count::numeric / NULLIF(h.count, 0) - 1) * 100)::numeric,
                0
              ), 2
            ) as variacao_percentual
        FROM recent r
        LEFT JOIN historical h ON r.classificacao = h.classificacao
        """)
        params = {"days_literal": f"{int(days)} days"}
        return pd.read_sql(query, self.engine, params=params)
    
    def get_low_confidence_predictions(self, threshold=0.6, limit=100):
        """Retorna predições com baixa confiança para revisão"""
        query = text("""
        SELECT * FROM logs_predicoes
        WHERE probabilidade < :threshold
        ORDER BY data DESC
        LIMIT :limit
        """)
        return pd.read_sql(query, self.engine, params={"threshold": float(threshold), "limit": int(limit)})
    
    def log_feedback(self, prediction_id, feedback, corrected_classification=None):
        """Registra feedback humano"""
        query = text("""
        UPDATE logs_predicoes
        SET feedback_humano = :feedback,
            classificacao = COALESCE(:corrigida, classificacao),
            data_feedback = NOW()
        WHERE id = :id
        """)
        # usar transaction/context manager para commit automático
        with self.engine.begin() as conn:
            conn.execute(query, {
                'id': int(prediction_id),
                'feedback': str(feedback),
                'corrigida': corrected_classification
            })
            conn.commit()