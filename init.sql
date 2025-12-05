-- 1. Cria a tabela da aplicação
CREATE TABLE IF NOT EXISTS reviews_features (
    review_id VARCHAR(50) PRIMARY KEY, -- identificador único do review
    texto_original TEXT, -- armazena o texto original do review
    texto_limpo TEXT, -- armazena o texto limpo do review
    target INT, -- armazena o target (0 ou 1)
    data_processamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Cria a tabela de curadoria de predições (feedback)
CREATE TABLE IF NOT EXISTS logs_predicoes (
    id SERIAL PRIMARY KEY, -- identificador único da predição
    data TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- armazena a data da predição
    texto_input TEXT NOT NULL, -- armazena o texto de entrada
    classificacao VARCHAR(50) NOT NULL, -- INSATISFEITO, SATISFEITO
    probabilidade FLOAT NOT NULL, -- armazena a probabilidade de ser INSATISFEITO
    probabilidade_confianca FLOAT, -- armazena a confiança da predição
    confianca_baixa BOOLEAN GENERATED ALWAYS AS (probabilidade_confianca < 0.6) STORED, -- armazena se a confiança é baixa
    modelo_version VARCHAR(50), -- armazena a versão do modelo usado
    tempo_inferencia_ms FLOAT, -- tempo de inferência em milissegundos
    feedback_humano VARCHAR(50), -- INSATISFEITO, SATISFEITO, CORRETO, INCORRETO
    data_feedback TIMESTAMP -- data do feedback humano
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