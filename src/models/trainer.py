import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

def train_sentiment_model(db_connection_str):
    """Lê do Postgres, treina modelo e loga no MLflow"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("sentiment_analysis")
    
    print("Lendo dados da Feature Store...")
    engine = create_engine(db_connection_str)
    df = pd.read_sql("SELECT * FROM reviews_features", engine)
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuração dos Modelos e Hiperparâmetros
    # Dicionário contendo o modelo e a grade de parâmetros para testar
    SEED = 42
    configuracoes = {
        "Logistic_Regression": {
            "modelo": LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],  # Força da regularização
                "penalty": ["l1", "l2"] # Tipo de penalidade
            }
        },
        "Random_Forest": {
            "modelo": RandomForestClassifier(class_weight='balanced', random_state=SEED),
            "params": {
                "n_estimators": [50, 100, 200],   # Número de árvores
                "max_depth": [10, 20, 30, None],  # Profundidade máxima
                "min_samples_leaf": [1, 2, 4]     # Ajuda a evitar overfitting em folhas muito específicas
            }
        },
        "XGBoost": {
            "modelo": XGBClassifier(eval_metric='logloss', scale_pos_weight=5, random_state=SEED),
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
    
    # Variáveis para rastrear o melhor modelo global
    best_global_f1 = -1
    best_global_run_id = ""
    best_global_model_name = ""
    
    # Loop de Treinamento com GridSearch
    for nome_modelo, config in configuracoes.items():
        with mlflow.start_run(run_name=nome_modelo) as run:
            print(f"Treinando {nome_modelo}...")

            # Configura o GridSearch
            # cv=3: Validação cruzada com 3 dobras (divide treino em 3 pedaços)
            # scoring='f1': Otimiza focado no F1-Score
            #grid = GridSearchCV(
            #    estimator=config["modelo"], 
            #    param_grid=config["params"], 
            #    cv=3, 
            #    scoring='f1',
            #    n_jobs=-1, # Usa todos os processadores para ser mais rápido
            #   verbose=1
            #)

            # Configura o RandomizedSearch
            random_search = RandomizedSearchCV(
                estimator=config["modelo"], 
                param_distributions=config["params"], 
                n_iter=25,  # Número de combinações a testar
                cv=3, 
                scoring='f1',
                n_jobs=-1,   # Usa todos os processadores para ser mais rápido
                verbose=1,
                random_state=SEED
            )
            
            # Treina testando todas as combinações
            with joblib.parallel_backend('threading', n_jobs=-1):
                # grid.fit(X_train, y_train)
                random_search.fit(X_train, y_train)

            # Pega o melhor modelo encontrado para esse algoritmo
            melhor_modelo_atual = random_search.best_estimator_
            melhores_params_atual = random_search.best_params_
            
            # Faz a previsão nos dados de TESTE
            preds = melhor_modelo_atual.predict(X_test)
            
            # Calcula Métricas
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds)
            }
            
            # Logar Parâmetros e Métricas
            mlflow.log_params(melhores_params_atual) # Loga os hiperparâmetros vencedores
            mlflow.log_metrics(metrics)
            
            # Logar Modelo e Vetorizador
            mlflow.sklearn.log_model(melhor_modelo_atual, "model")
            mlflow.sklearn.log_model(tfidf, "vectorizer")
            
            print(f"{nome_modelo} (Melhor Config) - F1: {metrics['f1']:.4f}")
            
            # Verifica se é o melhor modelo geral
            if metrics['f1'] > best_global_f1:
                best_global_f1 = metrics['f1']
                best_global_run_id = run.info.run_id
                best_global_model_name = nome_modelo

    # --- ETAPA DE REGISTRO (MODEL REGISTRY) ---
    if best_global_run_id:
        print(f"\n Melhor Modelo Geral: {best_global_model_name} com F1: {best_global_f1:.4f}")
        print(f"Registrando modelo do Run ID: {best_global_run_id} para Produção...")
        
        # Nome do modelo no Registry
        model_registry_name = "modelo_sentimento_bacen"
        model_uri = f"runs:/{best_global_run_id}/model"
        
        # 1. Registra a versão
        registered_model = mlflow.register_model(model_uri, model_registry_name)
        
        # 2. Promove para Produção (Stage='Production')
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_registry_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True # Move o anterior para Archived
        )
        print(f"Modelo versão {registered_model.version} promovido para PRODUCTION!")
        
    return f"Pipeline finalizado. Campeão: {best_global_model_name}"