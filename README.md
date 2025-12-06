# Projeto MLOps: Análise de Sentimento End-to-End

Este projeto implementa um sistema completo de MLOps para um modelo de análise de sentimento. Ele abrange desde o processamento inicial de dados e treinamento até o deploy de uma API de inferência, monitoramento contínuo e retreinamento automatizado.

## 📜 Visão Geral da Arquitetura

O sistema é orquestrado em contêineres Docker e utiliza uma arquitetura de microsserviços para garantir desacoplamento e escalabilidade.

```
      +-----------------+      +-----------------+      +-----------------+
      |  Frontend (UI)  |----->|   API (FastAPI) |----->|  MLflow Server  |
      |   (Streamlit)   |      +-----------------+      | (Model Registry)|
      +-----------------+               |               +-------+---------+
                                        |                       |
      +-----------------+               |                       |
      | Dashboard (UI)  |<--------------+-----------------------+------>+------------------+
      |   (Streamlit)   |               |                               |  App Database    |
      +-----------------+               |                               | (Postgres - App) |
                                        |                               | - Features       |
      +-----------------+               v                               | - Prediction Logs|
      | Airflow         |<------------>+------------------+              | - MLflow Backend |
      | - Webserver     |                                               +------------------+
      | - Scheduler     |
      | - Worker(s)     |
      +-----------------+
```

### Componentes Principais

*   **Airflow**: Orquestra os pipelines de dados e machine learning.
    *   **ETL Inicial**: Um pipeline para processar o dataset bruto, extrair features e treinar a primeira versão do modelo.
    *   **Ciclo de Vida do Modelo**: Uma DAG diária que monitora a performance do modelo em produção, detecta data drift, verifica novos feedbacks e dispara o retreinamento quando necessário.
*   **MLflow**: Centraliza o ciclo de vida do modelo.
    *   **Tracking**: Registra experimentos, parâmetros, métricas e artefatos de cada treinamento.
    *   **Model Registry**: Versiona os modelos e gerencia seus estágios (Staging, Production).
*   **FastAPI**: Fornece uma API RESTful para servir as predições do modelo em produção.
*   **PostgreSQL (x2)**:
    *   **`postgres`**: Banco de dados de metadados exclusivo para o Airflow.
    *   **`postgres_app` (aliás `postgres_bacen`)**: Banco de dados da aplicação, que armazena:
        *   `reviews_features`: Uma "Feature Store" simplificada com os dados processados para treinamento.
        *   `logs_predicoes`: Logs de todas as predições feitas pela API, incluindo feedbacks.
        *   *MLflow Backend*: Tabelas para armazenar os metadados de experimentos e modelos do MLflow.
*   **Streamlit (x2)**:
    *   **Frontend**: Uma interface de usuário simples para interagir com a API de predição.
    *   **Dashboard**: Um painel para monitorar a saúde do modelo, visualizar métricas e predições de baixa confiança.
*   **Celery & Redis**: Utilizados pelo Airflow para executar tarefas de forma distribuída e assíncrona.

## 🚀 Setup e Instalação

### Pré-requisitos
*   Docker
*   Docker Compose

### Passos para Instalação

1.  **Clone o Repositório**
    ```bash
    git clone <url-do-seu-repositorio>
    cd ML_projeto_final
    ```

2.  **Crie o Arquivo de Ambiente**
    Copie o arquivo de exemplo `.env.example` para `.env`.
    ```bash
    cp .env.example .env
    ```
    Abra o arquivo `.env` e gere uma chave Fernet para o Airflow:
    ```bash
    # No seu terminal, execute o comando abaixo e copie o resultado
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    ```
    Cole a chave gerada na variável `AIRFLOW_FERNET_KEY` dentro do arquivo `.env`.

    Seu arquivo .env deve conter as seguintes variáveis (ajuste conforme necessário):
    ```bash
    POSTGRES_USER=airflow
    POSTGRES_PASSWORD=airflow123
    POSTGRES_DB_APP=bacen
    MLFLOW_DB_USER=airflow
    MLFLOW_DB_PASSWORD=airflow123
    MLFLOW_DB_NAME=mlflow_db
    AIRFLOW_FERNET_KEY=<cole_a_chave_gerada_aqui>
    AIRFLOW_UID=50000
    ```

3.  **Defina o ID do Usuário Airflow**
    Para evitar problemas de permissão com os arquivos gerados pelo Airflow, defina o ID do seu usuário local.
    ```bash
    echo "AIRFLOW_UID=$(id -u)" >> .env
    ```

4.  **Inicie os Serviços**
    Execute o Docker Compose para construir as imagens e iniciar todos os contêineres em segundo plano.
    ```bash
    docker-compose up -d --build
    ```
    A primeira inicialização pode levar alguns minutos, pois o Airflow precisa inicializar seu banco de dados.

## ⚙️ Como Operar o Sistema

### Passo 1: Treinamento Inicial

O sistema começa sem nenhum modelo treinado. Você precisa executar o pipeline de ETL e treinamento inicial manualmente.

1.  Acesse a interface do Airflow: `http://localhost:8080` (usuário/senha: `airflow`/`airflow`).
2.  Encontre a DAG `sentiment_initial_etl_and_training`.
3.  Ative a DAG clicando no botão de toggle e, em seguida, clique no botão "Play" (▶️) para disparar uma execução.

Este processo irá:
*   Ler o dataset da Olist.
*   Processar e salvar as features no banco de dados da aplicação (tabela reviews_features).
*   Treinar múltiplos modelos de classificação (Logistic Regression / Random Forest / XGBoost / LinearSVC_Calibrated / LightGBM), selecionar o melhor, validá-lo e promovê-lo para "Production" no MLflow.

### Passo 2: Utilizando a API

Após o primeiro modelo ser treinado e promovido, a API estará pronta para servir predições.

*   **Documentação Interativa (Swagger)**: `http://localhost:8000/docs`
*   **Exemplo de Requisição `POST /predict`**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'Content-Type: application/json' \
      -d '{
      "message": "Consegui resolver o meu problema no chat. Atendimento super ágil!"
    }'
    ```

### Passo 3: Fornecendo Feedback

Cada predição retorna um `prediction_id`. Use este ID para registrar um feedback, que será usado no retreinamento.

*   **Exemplo de Requisição `POST /feedback/{prediction_id}`**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/feedback/1?feedback=INCORRETO&corrected_class=INSATISFEITO'
    ```

### Passo 4: Ciclo de Vida Automatizado

A DAG `sentiment_model_lifecycle` é executada automaticamente todos os dias. Ela:
1.  Verifica se a distribuição das predições recentes mudou (data drift).
2.  Verifica se há um número suficiente de novos feedbacks (padrão: 50).
3.  Se qualquer uma das condições for atendida, ela dispara o pipeline de retreinamento.
4.  O novo modelo treinado é comparado com o modelo em produção (padrão Champion-Challenger).
5.  Se o novo modelo for melhor, ele é automaticamente promovido para "Production", e a API passará a usá-lo na próxima reinicialização ou na próxima carga.

## 🌐 Acessando os Serviços

*   **Airflow UI**: `http://localhost:8080`
*   **Flower (Monitoramento Celery)**: `http://localhost:5555`
*   **MLflow UI**: `http://localhost:5000`
*   **API (Swagger UI)**: `http://localhost:8000/docs`
*   **Frontend App**: `http://localhost:8501`
*   **Dashboard de Monitoramento**: `http://localhost:8601`
*   **pgAdmin (Admin do Banco)**: `http://localhost:5050`

## 📂 Estrutura do Projeto

```
├── dags/                 # Definições das DAGs do Airflow
├── data/                 # Datasets brutos
├── docker/               # Dockerfiles para cada serviço
├── logs/                 # Logs do Airflow (mapeado do contêiner)
├── mlflow_artifacts/     # Artefatos do MLflow (modelos, etc.)
├── src/                  # Código-fonte da aplicação
│   ├── api/              # Lógica da API FastAPI
│   ├── db/               # Definições de schema e repositório do banco
│   ├── etl/              # Scripts para os pipelines de ETL e retreino
│   ├── frontend/         # Código do app Streamlit de interação
│   ├── models/           # Lógica de treinamento e avaliação de modelos
│   └── monitoring/       # Lógica para o dashboard e detecção de drift
├── .env                  # Variáveis de ambiente (secretas)
├── .env.example          # Arquivo de exemplo para o .env
├── docker-compose.yaml   # Orquestração de todos os serviços
└── README.md             # Esta documentação
```

## Rodando testes
```bash
pytest
```
Para rodar os testes de unidade do projeto, execute o comando a seguir na raiz do repositório:
```bash
pytest -q
```