import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  # Importação corrigida
from langchain.chains import RetrievalQA

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração do Streamlit
st.set_page_config(page_title="Agente LLM de Promoções", layout="wide")
st.title("Chat de Promoções com RAG")

# Carregar arquivo de produtos em promoção
FILE_PATH = "promocoes.txt"  # Certifique-se de que o arquivo está no mesmo diretório

@st.cache_resource
def load_data():
    if os.path.exists("db"):
        # Se o banco de dados já existe, carregue-o
        vectorstore = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
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

# Criar repositório vetorial
vectorstore = load_data()
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)  # Usando ChatOpenAI
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Pergunte sobre as promoções...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        response = qa_chain.run(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")