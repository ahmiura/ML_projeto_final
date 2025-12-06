import streamlit as st
import pandas as pd
import os
import plotly.express as px
from src.monitoring.service import PredictionMonitoring

st.set_page_config(page_title="Monitoramento", page_icon="📊", layout="wide")
st.title("📊 Monitoramento de Predições")

# --- Configurações e Conexão ---
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")
db_host = "postgres_app"
db_name = os.getenv("POSTGRES_DB_APP")
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"

# Instancia o serviço de monitoramento
@st.cache_resource
def get_monitoring_service():
    return PredictionMonitoring(DATABASE_URL)

monitoring_service = get_monitoring_service()

# --- Sidebar com Filtros ---
st.sidebar.header("Filtros")
days_to_filter = st.sidebar.selectbox(
    "Selecione o período de análise:",
    options=[7, 15, 30],
    index=0,
    format_func=lambda x: f"Últimos {x} dias"
)

# --- Carregamento de Dados com Cache ---
@st.cache_data(ttl=120)
def load_data(days):
    metrics_df = monitoring_service.get_metrics_by_period(days=days)
    drift_df = monitoring_service.detect_drift(days=days)
    low_confidence_df = monitoring_service.get_low_confidence_predictions(limit=100)
    return metrics_df, drift_df, low_confidence_df

df_metrics, df_drift, df_low_confidence = load_data(days_to_filter)

if df_metrics.empty:
    st.warning(f"⚠️ Nenhuma predição registrada nos últimos {days_to_filter} dias.")
else:
    # --- Métricas Principais ---
    st.subheader("Métricas de Negócio e Modelo")
    col1, col2, col3, col4 = st.columns(4, gap="large")
    total_predicoes = df_metrics['total_predicoes'].sum()
    total_insatisfeitos = df_metrics['insatisfeitos'].sum()
    total_baixa_confianca = df_metrics['baixa_confianca'].sum()

    with col1:
        st.metric("Total de Predições", f"{total_predicoes:,.0f}")

    with col2:
        taxa_insatisfeito = (total_insatisfeitos / total_predicoes) * 100 if total_predicoes > 0 else 0
        st.metric("Taxa de Insatisfação", f"{taxa_insatisfeito:.1f}%")

    with col3:
        prob_media_geral = (df_metrics['prob_media'] * df_metrics['total_predicoes']).sum() / total_predicoes if total_predicoes > 0 else 0
        st.metric("Prob. Média (Insatisf.)", f"{prob_media_geral:.2%}")

    with col4:
        st.metric("Predições Baixa Confiança", f"{total_baixa_confianca:,.0f}")

    st.markdown("---")

    # --- Métricas Operacionais ---
    st.subheader("Métricas Operacionais (Saúde da API)")
    col_op1, col_op2, col_op3 = st.columns(3, gap="large")

    # Calcula a média ponderada da latência para o período todo
    latencia_media_geral = (df_metrics['latencia_media_ms'] * df_metrics['total_predicoes']).sum() / total_predicoes if total_predicoes > 0 else 0
    # Para P95, a média dos P95 diários é uma aproximação. O ideal seria recalcular no período todo, mas isso é bom para o dashboard.
    latencia_p95_geral = df_metrics['latencia_p95_ms'].mean() if not df_metrics.empty else 0

    with col_op1:
        st.metric("Latência Média", f"{latencia_media_geral:.2f} ms")

    with col_op2:
        st.metric("Latência P95", f"{latencia_p95_geral:.2f} ms", help="95% das predições foram mais rápidas que este valor.")

    with col_op3:
        st.metric("Taxa de Erros API", "N/A", help="A taxa de erros da API (ex: HTTP 5xx) não é capturada aqui. Requer uma ferramenta de APM (Application Performance Monitoring) externa.")
    
    st.markdown("---")

    # --- Visualizações ---
    col_a, col_b = st.columns(2)

    with col_a:
        # Gráfico 1: Detecção de Data Drift
        st.subheader("1️⃣ Detecção de Data Drift")
        if not df_drift.empty:
            fig_drift = px.bar(
                df_drift,
                x='classificacao',
                y='variacao_percentual',
                title=f"Variação da Distribuição (Últimos {days_to_filter} dias vs. Histórico)",
                labels={"variacao_percentual": "Variação Percentual (%)", "classificacao": "Sentimento"},
                color='classificacao',
                color_discrete_map={"INSATISFEITO": "#ef553b", "SATISFEITO": "#00cc96"},
                text_auto='.2f'
            )
            fig_drift.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_drift, use_container_width=True)
            st.info("""
            Este gráfico compara a proporção de cada sentimento no período recente com um período histórico.
            **Variações grandes (ex: > 20%) podem indicar Data Drift**, sugerindo a necessidade de investigar os dados ou retreinar o modelo.
            """)
        else:
            st.info("Não há dados suficientes para calcular o drift.")

    with col_b:
        # Gráfico 2: Evolução Temporal
        st.subheader("2️⃣ Predições por Dia")
        fig_tempo = px.bar(
            df_metrics.sort_values('dia'),
            x='dia',
            y=['total_predicoes', 'insatisfeitos'],
            title="Volume de Predições e Insatisfeitos por Dia",
            labels={"dia": "Data", "value": "Quantidade"},
            barmode='group'
        )
        st.plotly_chart(fig_tempo, use_container_width=True)

    st.markdown("---")

    # Gráfico 3: Distribuição de Probabilidades
    st.subheader("3️⃣ Distribuição de Confiança das Predições")
    # Para este gráfico, ainda precisamos de dados mais granulares.
    # Vamos carregar apenas as colunas necessárias para otimizar.
    @st.cache_data(ttl=120)
    def load_probabilities(days):
        query = "SELECT probabilidade_confianca FROM logs_predicoes WHERE data >= NOW() - INTERVAL '{} days'".format(int(days))
        return pd.read_sql(query, DATABASE_URL)

    df_probs = load_probabilities(days_to_filter)
    fig_prob = px.histogram(
        df_probs,
        x='probabilidade_confianca',
        nbins=25,
        title="Histograma da Confiança da Predição",
        labels={"probabilidade_confianca": "Nível de Confiança"},
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # Tabela de Predições com Baixa Confiança para Revisão
    st.subheader("4️⃣ Revisão: Predições com Baixa Confiança")
    if not df_low_confidence.empty:
        st.warning(f"🔴 Encontradas {len(df_low_confidence)} predições com confiança abaixo do limiar para revisão.")
        st.dataframe(df_low_confidence, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Nenhuma predição recente com baixa confiança.")