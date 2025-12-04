import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Monitoramento", page_icon="📊", layout="wide")
st.title("📊 Monitoramento de Predições")

# --- Configurações lidas do ambiente (via .env) ---
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")
db_host = "postgres_app"
db_name = os.getenv("POSTGRES_DB_APP")

# Conexão com Banco
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
engine = create_engine(DATABASE_URL)

# Query os dados de logs
@st.cache_data(ttl=60)
def load_predictions():
    """Carrega dados dos últimos 30 dias"""
    query = """
    SELECT * FROM logs_predicoes 
    WHERE data >= NOW() - INTERVAL '30 days'
    ORDER BY data DESC
    """
    return pd.read_sql(query, engine)

df_logs = load_predictions()

if df_logs.empty:
    st.warning("⚠️ Nenhuma predição registrada ainda.")
else:
    # Métricas Principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Predições", len(df_logs))
    
    with col2:
        taxa_insatisfeito = (df_logs['classificacao'] == 'INSATISFEITO').sum() / len(df_logs) * 100
        st.metric("Taxa de Insatisfação", f"{taxa_insatisfeito:.1f}%")
    
    with col3:
        prob_media = df_logs['probabilidade'].mean()
        st.metric("Probabilidade Média", f"{prob_media:.2%}")
    
    with col4:
        dias_ativo = (datetime.now() - df_logs['data'].min()).days
        st.metric("Dias em Produção", dias_ativo)
    
    # Gráfico 1: Distribuição de Sentimentos
    st.subheader("1️⃣ Distribuição de Sentimentos")
    fig_sentimentos = px.pie(
        df_logs, 
        names='classificacao',
        title="Proporção de Sentimentos",
        color_discrete_map={"INSATISFEITO": "#ef553b", "SATISFEITO": "#00cc96"}
    )
    st.plotly_chart(fig_sentimentos, use_container_width=True)
    
    # Gráfico 2: Evolução Temporal
    st.subheader("2️⃣ Predições por Hora")
    df_hora = df_logs.set_index('data').resample('1H').size().reset_index(name='count')
    fig_tempo = px.bar(
        df_hora,
        x='data',
        y='count',
        title="Número de Predições por Hora",
        labels={"data": "Hora", "count": "Predições"}
    )
    st.plotly_chart(fig_tempo, use_container_width=True)
    
    # Gráfico 3: Distribuição de Probabilidades
    st.subheader("3️⃣ Distribuição de Confiança")
    fig_prob = px.histogram(
        df_logs,
        x='probabilidade',
        nbins=20,
        title="Histograma de Probabilidades",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Tabela de Últimas Predições
    st.subheader("4️⃣ Últimas Predições")
    st.dataframe(
        df_logs.head(20),
        use_container_width=True,
        hide_index=True
    )
    
    # Detecção de Anomalias (Baixa Confiança)
    st.subheader("⚠️ Predições com Baixa Confiança (< 60%)")
    baixa_confianca = df_logs[df_logs['probabilidade'] < 0.6]
    if len(baixa_confianca) > 0:
        st.warning(f"🔴 {len(baixa_confianca)} predições com confiança baixa!")
        st.dataframe(baixa_confianca, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Nenhuma predição com baixa confiança nos últimos 30 dias!")