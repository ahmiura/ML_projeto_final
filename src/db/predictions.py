# src/db/predictions.py
from sqlalchemy import Table, Column, MetaData, Integer, Text, Float, TIMESTAMP, create_engine
from sqlalchemy import insert, select, update
from datetime import datetime

metadata = MetaData()

logs_predicoes = Table(
    "logs_predicoes",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("data", TIMESTAMP),
    Column("texto_input", Text),
    Column("classificacao", Text),
    Column("probabilidade", Float),
    Column("probabilidade_confianca", Float),
    Column("tempo_inferencia_ms", Float),
    Column("modelo_version", Text),
    Column("feedback_humano", Text),
    Column("data_feedback", TIMESTAMP),
)

class PredictionsRepo:
    def __init__(self, engine):
        self.engine = engine

    def create_tables(self):
        metadata.create_all(self.engine)

    def insert_prediction(self, texto, classificacao, prob, confianca, tempo_ms, modelo_version) -> int:
        stmt = insert(logs_predicoes).values(
            data=datetime.utcnow(),
            texto_input=texto,
            classificacao=classificacao,
            probabilidade=prob,
            probabilidade_confianca=confianca,
            tempo_inferencia_ms=tempo_ms,
            modelo_version=modelo_version
        ).returning(logs_predicoes.c.id)
        with self.engine.connect() as conn:
            res = conn.execute(stmt)
            conn.commit()
            return int(res.scalar())

    def get_low_confidence(self, threshold=0.6, limit=100):
        stmt = select(logs_predicoes).where(logs_predicoes.c.probabilidade_confianca < threshold).order_by(logs_predicoes.c.data.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).fetchall()]

    def get_feedback_data(self):
        """Busca todos os registros que possuem feedback humano para retreinamento."""
        stmt = select(logs_predicoes.c.texto_input, logs_predicoes.c.classificacao).where(logs_predicoes.c.feedback_humano.isnot(None))
        with self.engine.connect() as conn:
            return conn.execute(stmt).fetchall()

    def update_feedback(self, prediction_id: int, feedback: str, corrected_class: str = None):
        stmt = update(logs_predicoes).where(logs_predicoes.c.id == prediction_id).values(
            feedback_humano=feedback,
            classificacao=corrected_class if corrected_class else logs_predicoes.c.classificacao,
            data_feedback=datetime.utcnow()
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)