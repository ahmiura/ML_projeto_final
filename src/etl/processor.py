import pandas as pd
from sqlalchemy import create_engine
import os
import re
from unidecode import unidecode

def clean_text(text):
    """Função para limpar o texto: remove acentos, converte para minúsculas, remove pontuação e espaços extras."""
    if not isinstance(text, str):
        return ""
    text = unidecode(text) # Remove acentos (ex: "péssimo" -> "pessimo")
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'\d+', ' ', text)  # Remove números, substitui por espaço
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove números e pontuação, substitui por espaço
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    text = ' '.join(text.split()) # Remove espaços extras
    return text

def extract_data(input_path, output_path):
    """ETAPA 1: Extração (Lê do CSV bruto e salva em Parquet temporário)"""
    print(f"--- [EXTRACT] Lendo dados de {input_path} ---")
    df = pd.read_csv(input_path)
    
    # Apenas verificações básicas de leitura
    print(f"Dados extraídos: {df.shape}")
    
    # Salva no formato (Parquet) para a próxima etapa
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dados brutos salvos em {output_path}")
    return output_path

def transform_data(input_path, output_path):
    """ETAPA 2: Transformação (Lê bruto, aplica limpeza e regras de negócio)"""
    print(f"--- [TRANSFORM] Processando dados de {input_path} ---")
    df = pd.read_parquet(input_path)
    
    # 1. Filtrar nulos
    df = df.dropna(subset=['review_comment_message', 'review_id'])
    
    # 2. Criar Target (Regra de Negócio) se review_score < 3 => 1 (negativo), else 0 (positivo)
    df['target'] = df['review_score'].apply(lambda x: 1 if x < 3 else 0)
    
    # 3. Limpeza de texto usando a função dedicada
    df['texto_limpo'] = df['review_comment_message'].apply(clean_text)
    
    # Selecionar colunas finais
    df_final = df[['review_id', 'review_comment_message', 'texto_limpo', 'target']].copy()
    df_final.rename(columns={'review_comment_message': 'texto_original'}, inplace=True)
    
    print(f"Dados transformados: {df_final.shape}")
    df_final.to_parquet(output_path, index=False)
    print(f"Dados limpos salvos em {output_path}")
    return output_path

def load_data(input_path, db_connection_str, table_name='reviews_features'):
    """ETAPA 3: Carga (Lê dados limpos e salva no Postgres)"""
    print(f"--- [LOAD] Salvando dados de {input_path} no Banco ---")
    df = pd.read_parquet(input_path)
    
    engine = create_engine(db_connection_str)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    
    print(f"Sucesso! {len(df)} registros inseridos na tabela '{table_name}'.")