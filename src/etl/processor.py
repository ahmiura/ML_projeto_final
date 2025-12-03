"""Módulo para o pipeline de ETL de reviews."""

import logging
import os
import re
from typing import Optional

import pandas as pd
from unidecode import unidecode
from sqlalchemy import create_engine

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
# Nomes de colunas para evitar "magic strings"
COL_REVIEW_ID = 'review_id'
COL_COMMENT = 'review_comment_message'
COL_SCORE = 'review_score'
COL_TARGET = 'target'
COL_TEXTO_LIMPO = 'texto_limpo'
COL_TEXTO_ORIGINAL = 'texto_original'

# Limite para classificação de sentimento
TARGET_SCORE_THRESHOLD = 3


def clean_text(text: Optional[str]) -> str:
    """
    Limpa uma string de texto.

    - Remove acentos.
    - Converte para minúsculas.
    - Remove números e pontuação.
    - Remove espaços extras.

    Args:
        text: A string de entrada a ser limpa.

    Returns:
        A string limpa ou uma string vazia se a entrada for None ou não-string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove acentos e converte para minúsculas
    processed_text = unidecode(text).lower()
    # 2. Remove tudo que não for letra (a-z) ou espaço
    processed_text = re.sub(r'[^a-z\s]', '', processed_text)
    # 3. Remove espaços em branco extras entre palavras
    processed_text = ' '.join(processed_text.split())
    
    return processed_text

def extract_data(input_path: str, output_path: str) -> str:
    """ETAPA 1: Extração (Lê do CSV bruto e salva em Parquet temporário)"""
    logging.info(f"--- [EXTRACT] Lendo dados de {input_path} ---")
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Dados extraídos com sucesso: {df.shape}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logging.info(f"Dados brutos salvos em {output_path}")
    except FileNotFoundError:
        logging.error(f"Arquivo de entrada não encontrado em: {input_path}")
        raise
    return output_path

def transform_data(input_path: str, output_path: str) -> str:
    """ETAPA 2: Transformação (Lê bruto, aplica limpeza e regras de negócio)"""
    logging.info(f"--- [TRANSFORM] Processando dados de {input_path} ---")
    df = pd.read_parquet(input_path)
    
    # 1. Filtrar nulos
    df.dropna(subset=[COL_COMMENT, COL_REVIEW_ID], inplace=True)
    
    # 2. Criar Target (Regra de Negócio) se review_score < 3 => 1 (negativo), else 0 (positivo)
    df[COL_TARGET] = df[COL_SCORE].apply(lambda x: 1 if x < TARGET_SCORE_THRESHOLD else 0)
    
    # 3. Limpeza de texto usando a função dedicada
    df[COL_TEXTO_LIMPO] = df[COL_COMMENT].apply(clean_text)
    
    # Selecionar colunas finais
    df_final = df[[COL_REVIEW_ID, COL_COMMENT, COL_TEXTO_LIMPO, COL_TARGET]].copy()
    df_final.rename(columns={COL_COMMENT: COL_TEXTO_ORIGINAL}, inplace=True)
    
    logging.info(f"Dados transformados: {df_final.shape}")
    df_final.to_parquet(output_path, index=False)
    logging.info(f"Dados limpos salvos em {output_path}")
    return output_path

def load_data(input_path: str, db_connection_str: str, table_name: str = 'reviews_features'):
    """ETAPA 3: Carga (Lê dados limpos e salva no Postgres)"""
    logging.info(f"--- [LOAD] Salvando dados de {input_path} no Banco ---")
    df = pd.read_parquet(input_path)
    
    try:
        engine = create_engine(db_connection_str)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f"Sucesso! {len(df)} registros inseridos na tabela '{table_name}'.")
    except Exception as e:
        logging.error(f"Falha ao carregar dados no banco de dados: {e}")
        raise