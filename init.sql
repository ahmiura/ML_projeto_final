-- 1. Cria a tabela da aplicação
CREATE TABLE IF NOT EXISTS reviews_features (
    review_id VARCHAR(50) PRIMARY KEY,
    texto_original TEXT,
    texto_limpo TEXT,
    target INT,
    data_processamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Cria a tabela de curadoria de predições (feedback)
CREATE TABLE IF NOT EXISTS logs_predicoes (
    id SERIAL PRIMARY KEY,
    data TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    texto_input TEXT NOT NULL,
    classificacao VARCHAR(50) NOT NULL,
    probabilidade FLOAT NOT NULL,
    confianca_baixa BOOLEAN GENERATED ALWAYS AS (probabilidade < 0.6) STORED,
    modelo_version VARCHAR(50),
    tempo_inferencia_ms FLOAT,
    feedback_humano VARCHAR(50), -- INSATISFEITO, SATISFEITO, CORRETO, INCORRETO
    data_feedback TIMESTAMP
);

-- criar índices (Postgres)
CREATE INDEX IF NOT EXISTS idx_logs_predicoes_data ON logs_predicoes (data);
CREATE INDEX IF NOT EXISTS idx_logs_predicoes_classificacao ON logs_predicoes (classificacao);
CREATE INDEX IF NOT EXISTS idx_logs_predicoes_confianca ON logs_predicoes (confianca_baixa);

-- 3. Habilita dblink para poder criar outro banco
CREATE EXTENSION IF NOT EXISTS dblink;

-- 4. Cria o banco mlflow_db se não existir
DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db') THEN
      PERFORM dblink_exec('dbname=postgres', 'CREATE DATABASE mlflow_db OWNER airflow');
   END IF;
END
$do$;

-- 5. Garante permissões para o usuário airflow no banco mlflow_db
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO airflow;