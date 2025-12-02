import streamlit as st
import requests
import os

st.set_page_config(page_title="Análise de Sentimento", page_icon="🤖", layout="centered")

st.title("🤖 Análise de Sentimento de Clientes")
st.markdown("Simulador de um chatbot que decide se deve ou não transferir para um atendente humano com base no sentimento do cliente.")

# URL da API (Pega da variável de ambiente do Docker ou usa localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000")

texto = st.text_area("Digite a mensagem do cliente aqui:", height=150, placeholder="Ex: O app travou e sumiu meu dinheiro!")

if st.button("Analisar Sentimento"):
    if not texto:
        st.warning("Por favor, digite uma mensagem.")
    else:
        with st.spinner("Consultando modelo de Inteligência Artificial..."):
            try:
                # CORREÇÃO 1: Usar a chave 'message' que a API espera
                response = requests.post(f"{API_URL}/predict", json={"message": texto})
                
                if response.status_code == 200:
                    dados = response.json()
                    
                    # CORREÇÃO 2: Ler as chaves corretas que a API retorna
                    classe = dados['sentimento']                  # Antes era 'classificacao'
                    prob = dados['probabilidade_insatisfeito']    # Antes era 'probabilidade_risco'
                    acao = dados['acao_sugerida']
                    
                    # Exibição dos resultados
                    col1, col2 = st.columns(2)
                    col1.metric("Classificação", classe)
                    col1.metric("Probabilidade de Insatisfação", f"{prob:.2%}")
                    
                    # Lógica de cores baseada no valor correto ('INSATISFEITO')
                    if classe == 'INSATISFEITO':
                        st.error(f"🚨 ALERTA: Cliente com alto risco de reclamação no Bacen!")
                        st.error(f"Ação Recomendada: {acao}")
                    else:
                        st.success(f"✅ Cliente Satisfeito.")
                        st.info(f"Ação Recomendada: {acao}")
                        
                    with st.expander("Ver JSON da API"):
                        st.json(dados)
                else:
                    st.error(f"Erro na API: {response.status_code}")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"Erro de conexão: {e}")
                st.info(f"Tentando conectar em: {API_URL}/predict")