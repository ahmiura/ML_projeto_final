import streamlit as st
import requests
import os

st.set_page_config(page_title="Análise de Sentimento", page_icon="🤖", layout="centered")

st.title("🤖 Análise de Sentimento de Clientes")
st.markdown("Simulador de um chatbot que decide se deve ou não transferir para um atendente humano com base no sentimento do cliente.")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Inicializa estado da sessão
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

texto = st.text_area("Digite a mensagem do cliente aqui:", height=150, placeholder="Ex: O app travou e sumiu meu dinheiro!")

if st.button("Analisar Sentimento"):
    if not texto:
        st.warning("Por favor, digite uma mensagem.")
    else:
        with st.spinner("Consultando modelo de Inteligência Artificial..."):
            try:
                response = requests.post(f"{API_URL}/predict", json={"message": texto})
                
                if response.status_code == 200:
                    st.session_state["resultado"] = response.json()
                    # Guarda o texto original caso precise reenviar
                    st.session_state["texto_analisado"] = texto 
                else:
                    st.error(f"Erro na API: {response.status_code}")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

# --- EXIBIÇÃO DO RESULTADO E FEEDBACK ---
if st.session_state["resultado"]:
    dados = st.session_state["resultado"]
    
    classe = dados['sentimento']
    prob = dados['probabilidade_insatisfeito']
    acao = dados['acao_sugerida']
    pred_id = dados.get('prediction_id')
    
    st.write("---")
    col1, col2 = st.columns(2)
    col1.metric("Classificação", classe)
    col1.metric("Probabilidade de Insatisfação", f"{prob:.2%}")
    
    if classe == 'INSATISFEITO':
        st.error(f"🚨 ALERTA: Risco de Bacen! Ação: {acao}")
    else:
        st.success(f"✅ Cliente Seguro. Ação: {acao}")
    
    # --- ÁREA DE FEEDBACK ---
    if pred_id:
        st.write("### 📝 O modelo acertou?")
        
        # Botões lado a lado
        fb_col1, fb_col2 = st.columns(2)
        
        with fb_col1:
            if st.button("👍 Sim, acertou!"):
                try:
                    # Envia feedback CORRETO
                    requests.post(
                        f"{API_URL}/feedback/{pred_id}", 
                        params={"feedback": "CORRETO"}
                    )
                    st.toast("Feedback positivo registrado!", icon="✅")
                except Exception as e:
                    st.error(f"Erro ao enviar feedback: {e}")

        with fb_col2:
            if st.button("👎 Não, errou"):
                st.session_state["mostrar_correcao"] = True
        
        # Se clicou em "Errou", mostra opção de corrigir
        if st.session_state.get("mostrar_correcao"):
            opcao_correta = st.radio("Qual seria a classificação correta?", 
                                     ["SATISFEITO", "INSATISFEITO"], 
                                     horizontal=True)
            
            if st.button("Enviar Correção"):
                try:
                    # Envia feedback INCORRETO com a classe certa
                    requests.post(
                        f"{API_URL}/feedback/{pred_id}", 
                        params={"feedback": "INCORRETO", "corrected_class": opcao_correta}
                    )
                    st.success("Obrigado! O modelo aprenderá com seu feedback.")
                    st.session_state["mostrar_correcao"] = False
                except Exception as e:
                    st.error(f"Erro ao enviar correção: {e}")