# Pipeline de MLOps: Análise de Sentimento de E-commerce

Este projeto implementa uma plataforma completa de **MLOps (Machine Learning Operations)** para análise de sentimento de avaliações de clientes.

A solução utiliza uma arquitetura de microserviços para garantir escalabilidade, reprodutibilidade e monitoramento contínuo do ciclo de vida do modelo de Machine Learning.

## 🏛️ Arquitetura do Projeto

O sistema foi desenhado seguindo o padrão de **Monorepo Modular**, onde o código de negócio é compartilhado entre os serviços de treinamento e inferência para evitar discrepância de dados (*Training-Serving Skew*).

### Componentes Principais (Docker Containers)

1.  **Airflow (Orquestrador):** Gerencia o pipeline de dados (ETL) e o retreinamento periódico dos modelos.
    * *Executor:* Celery (Distribuído) com Redis.
2.  **PostgreSQL (Feature Store & Metadados):**
    * Armazena os dados tratados (`reviews_features`) prontos para treinamento.
    * Serve como backend para o Airflow e MLflow.
3.  **MLflow (Model Registry):**
    * Rastreia experimentos (métricas, parâmetros).
    * Gerencia o versionamento dos modelos e promove o melhor (F1-score) para "Produção".
4.  **API (Serving):**
    * Serviço FastAPI.
    * Carrega automaticamente a versão de produção do modelo do MLflow.
5.  **Frontend:**
    * Aplicação Streamlit para interação com o usuário e teste do modelo em tempo real.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* Docker e Docker Compose instalados.
* Git.

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/ml_projeto_final.git](https://github.com/seu-usuario/ml_projeto_final.git)
    cd ml_projeto_final
    ```

2.  **Configure as Credenciais:**
    Crie um arquivo `.env` na raiz do projeto (baseado no exemplo abaixo) para definir as senhas do banco de dados e chaves de segurança.

    ```env
    POSTGRES_USER=airflow
    POSTGRES_PASSWORD=airflow123
    POSTGRES_DB_APP=bacen
    
    MLFLOW_DB_USER=airflow
    MLFLOW_DB_PASSWORD=airflow123
    MLFLOW_DB_NAME=mlflow_db
    
    # Gere uma chave Fernet válida para o Airflow
    AIRFLOW_FERNET_KEY=SuaChaveGeradaAqui...
    AIRFLOW_UID=50000
    ```

3.  **Construa e Inicie os Serviços:**
    ```bash
    # 1. Construir as imagens Docker (pode levar alguns minutos)
    docker-compose build

    # 2. Inicializar o banco de dados do Airflow
    docker-compose up airflow-init

    # 3. Subir todo o ambiente em background
    docker-compose up -d
    ```

4.  **Acesse os Serviços:**

    | Serviço | URL | Credenciais (Padrão) |
    | :--- | :--- | :--- |
    | **Airflow** | [http://localhost:8080](http://localhost:8080) | `airflow` / `airflow` |
    | **MLflow** | [http://localhost:5000](http://localhost:5000) | - |
    | **API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | - |
    | **Frontend** | [http://localhost:8501](http://localhost:8501) | - |
    | **PgAdmin** | [http://localhost:5050](http://localhost:5050) | `admin@admin.com` / `admin` |

---

## 🧪 Executando o Pipeline

1.  Acesse o **Airflow** (`localhost:8080`).
2.  Ative o DAG **`olist_sentiment_pipeline`**.
3.  O pipeline executará automaticamente as etapas:
    * **Extração:** Lê o dataset bruto (`data/olist_order_reviews_dataset.csv`).
    * **Transformação:** Limpa o texto e cria features.
    * **Carga:** Salva os dados processados no PostgreSQL.
    * **Treinamento:** Treina múltiplos modelos (Logistic Regression, Random Forest, XGBoost), compara a performance e registra o vencedor no MLflow.

## 📊 Monitoramento e Melhoria Contínua

* **MLflow:** Acesse para ver o histórico de treinamentos, comparar a acurácia dos modelos e ver qual algoritmo venceu a batalha.
* **API:** Reinicie a API (`docker-compose restart api`) após um novo treinamento para que ela carregue automaticamente a nova versão do modelo campeão promovido a produção.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Machine Learning:** Scikit-Learn, XGBoost
* **Engenharia de Dados:** Pandas, SQLAlchemy
* **Infraestrutura:** Docker, Docker Compose, Redis
* **MLOps:** Apache Airflow, MLflow
* **Web:** FastAPI, Streamlit

---