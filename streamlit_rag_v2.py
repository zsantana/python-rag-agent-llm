import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import requests

# Configuração do Streamlit
st.set_page_config(page_title="Agente LLM de Promoções", layout="wide")
st.title("Chat de Promoções com RAG")

# Campo para API Key no topo da tela
OPENAI_API_KEY = st.sidebar.text_input("Insira sua chave da API OpenAI", type="password")

# Função para verificar se a chave API OpenAI é válida
def verificar_chave_api(api_key):
    try:
        # Tentativa de fazer uma chamada simples à API para verificar a chave
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers
        )
        return response.status_code == 200
    except Exception:
        return False

# Verificar se a chave foi fornecida
if not OPENAI_API_KEY:
    st.sidebar.warning("Por favor, insira sua chave da API OpenAI")
    st.stop()
else:
    # Verificar se a chave é válida
    chave_valida = verificar_chave_api(OPENAI_API_KEY)
    if chave_valida:
        st.sidebar.success("✅ Chave API OpenAI válida!")
    else:
        st.sidebar.error("❌ Chave API OpenAI inválida! Por favor, verifique e tente novamente.")
        st.stop()

# Carregar variáveis de ambiente
load_dotenv()

# Carregar arquivo de produtos em promoção
FILE_PATH = "promocoes.txt"  # Certifique-se de que o arquivo está no mesmo diretório

@st.cache_resource
def load_data():
    if os.path.exists("db"):
        # Se o banco de dados já existe, carregue-o
        vectorstore = Chroma(
            persist_directory="db", 
            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        )
    else:
        # Caso contrário, crie um novo banco de dados
        loader = TextLoader(FILE_PATH)
        documents = loader.load()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            persist_directory="db",
        )
    return vectorstore

# Criar repositório vetorial apenas se a chave for válida
if chave_valida:
    try:
        vectorstore = load_data()
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        
        # Interface de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_input = st.chat_input("Pergunte sobre as promoções...")
        
        if user_input:
            try:
                # Exibe a mensagem do usuário na interface
                with st.chat_message("user"):
                    st.write(user_input)
                    
                response = qa_chain.invoke(user_input)
                if isinstance(response, dict) and "result" in response:
                    response = response["result"]  # Extraindo apenas o valor relevante
                    
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                with st.chat_message("assistant"):
                    st.write(response)
                    
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")