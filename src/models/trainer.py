import pandas as pd
import logging
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42

def get_data(db_connection_str: str) -> pd.DataFrame:
    """Lê os dados da tabela de features no banco de dados."""
    logging.info("Lendo dados da Feature Store...")
    engine = create_engine(db_connection_str)
    try:
        df = pd.read_sql("SELECT texto_limpo, target FROM reviews_features", engine)
        logging.info(f"Dados carregados com sucesso: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Falha ao ler dados do banco: {e}")
        raise

def preprocess_data(df: pd.DataFrame):
    """Vetoriza o texto e divide os dados em treino e teste."""
    logging.info("Pré-processando e vetorizando dados...")
    # Vetorização - Otimizações
    tfidf = TfidfVectorizer(
        max_features=3000,       # Aumentado para capturar mais vocabulário
        ngram_range=(1, 2),      # Captura unigramas e bigramas (ex: "nao gostei")
        min_df=5,                # Ignora palavras que aparecem em menos de 5 reviews (ruído)
        max_df=0.9,              # Ignora palavras que aparecem em mais de 90% dos reviews
        sublinear_tf=True        # Aplica log na contagem (suaviza palavras muito repetidas)
    )
    X = tfidf.fit_transform(df['texto_limpo'])
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    return X_train, X_test, y_train, y_test, tfidf

def get_model_configs():
    configuracoes = {
        "Logistic_Regression": {
            "modelo": LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],  # Força da regularização
                "penalty": ["l1", "l2"] # Tipo de penalidade
            }
        },
        "Random_Forest": {
            "modelo": RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100, 200],   # Número de árvores
                "max_depth": [10, 20, 30, None],  # Profundidade máxima
                "min_samples_leaf": [1, 2, 4]     # Ajuda a evitar overfitting em folhas muito específicas
            }
        },
        "XGBoost": {
            "modelo": XGBClassifier(
                eval_metric='logloss', 
                scale_pos_weight=5, 
                random_state=SEED,
                # Otimização de performance: 'hist' é muito mais rápido que o método 'exact' padrão
                tree_method='hist',
                n_jobs=-1
            ),
            "params": {
                "learning_rate": [0.01, 0.1, 0.2],  # Taxa de aprendizado
                "n_estimators": [100, 200],         # Número de árvores
                "max_depth": [3, 6, 10],            # Profundidade da árvore
                "subsample": [0.8, 1.0],            # Treina com apenas 80% dos dados por árvore (evita overfitting)
                "colsample_bytree": [0.8, 1.0]      # Usa apenas 80% das colunas (palavras) por árvore
            }
        },
        "LinearSVC_Calibrated": {
            # LinearSVC não tem predict_proba -> calibramos para obter probabilidades
            "modelo": CalibratedClassifierCV(LinearSVC(random_state=SEED, dual=False, max_iter=5000), cv=3),
            "params": {
                # Para acessar o estimador interno usamos 'estimator__' prefixado
                "estimator__C": [0.01, 0.1, 1.0]
            }
        },
        "LightGBM": {
            "modelo": lgb.LGBMClassifier(random_state=SEED, n_jobs=-1),
            "params": {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63]
            }
        }
    }
    return configuracoes

def train_and_evaluate_models(X_train, X_test, y_train, y_test, tfidf):
    """Itera sobre as configurações, treina, avalia e loga os modelos no MLflow."""
    configuracoes = get_model_configs()
    best_run = {"f1": -1, "run_id": None, "model_name": None}

    for nome_modelo, config in configuracoes.items():
        with mlflow.start_run(run_name=nome_modelo) as run:
            logging.info(f"Treinando {nome_modelo}...")

            random_search = RandomizedSearchCV(
                estimator=config["modelo"], 
                param_distributions=config["params"], 
                n_iter=20,  # Número de combinações a testar
                cv=3, 
                scoring='f1',
                n_jobs=-1,   # Usa todos os processadores para ser mais rápido
                verbose=1,
                random_state=SEED
            )
            
            with joblib.parallel_backend('threading', n_jobs=-1):
                random_search.fit(X_train, y_train)
            
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            
            preds = best_model.predict(X_test)
            metrics = evaluate_model(y_test, preds)
            
            # Logar Parâmetros e Métricas
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            
            # Logar Modelo e Vetorizador
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.sklearn.log_model(tfidf, "vectorizer")
            
            logging.info(f"{nome_modelo} (Melhor Config) - F1: {metrics['f1']:.4f}")
            
            # Verifica se é o melhor modelo geral
            if metrics['f1'] > best_run["f1"]:
                best_run["f1"] = metrics['f1']
                best_run["run_id"] = run.info.run_id
                best_run["model_name"] = nome_modelo
    return best_run

def evaluate_model(y_true, y_pred):
    """Calcula e retorna um dicionário de métricas."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def register_best_model(best_run: dict, registry_name: str = "modelo_sentimento_bacen"):
    """Registra o melhor modelo do experimento no Model Registry e o promove para Produção."""
    if best_run["run_id"]:
        logging.info(f"\nMelhor Modelo Geral: {best_run['model_name']} com F1: {best_run['f1']:.4f}")
        logging.info(f"Registrando modelo do Run ID: {best_run['run_id']} para Produção...")
        
        model_uri = f"runs:/{best_run['run_id']}/model"
        registered_model = mlflow.register_model(model_uri, registry_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=registry_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        logging.info(f"Modelo versão {registered_model.version} promovido para PRODUCTION!")

def train_sentiment_model(db_connection_str: str):
    """Lê do Postgres, treina modelo e loga no MLflow"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("sentiment_analysis")

    df = get_data(db_connection_str)
    X_train, X_test, y_train, y_test, tfidf = preprocess_data(df)
    best_run = train_and_evaluate_models(X_train, X_test, y_train, y_test, tfidf)
    register_best_model(best_run)
    
    return f"Pipeline finalizado. Campeão: {best_run['model_name']}"