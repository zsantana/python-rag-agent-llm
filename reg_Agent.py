import os
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Configuração do logging
logging.basicConfig(
    level=logging.DEBUG,  # Define o nível de log para DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Carrega variáveis do ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Função para carregar o texto de um arquivo externo
def carregar_texto(caminho_arquivo: str) -> str:
    """Lê o conteúdo de um arquivo .txt e retorna como string."""
    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
            return arquivo.read()
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        exit(1)
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo: {e}")
        exit(1)

# Defina o caminho do arquivo .txt com o texto
CAMINHO_ARQUIVO = "texto.txt"  # Altere para o nome correto do arquivo
TEXTO = carregar_texto(CAMINHO_ARQUIVO)

# Criando o documento
documents = [Document(page_content=TEXTO, metadata={"autor": "Anwar", "source": "https://medium.com/@anwarhermuche/apenas-tr%C3%AAs-quest%C3%B5es-eb2cce38a6f6"})]

# Divisão do texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

# Criando o banco vetorial
db = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    persist_directory="db",
)

# Definição do template para RAG
template = PromptTemplate.from_template("""
Answer the user query based on context. If you don't know the answer or the context does not have the answer, say that you don't know.
ALWAYS answer in pt-BR. You are going to answer questions about Anwar's life. Suppose you're him. Answer the questions with "Eu fiz isso...", "Olha, eu sou um cara..."
ALWAYS answer with the source of your knowledge. Provide a URL.

## CONTEXT
{contexto}

## USER QUERY
{pergunta}
""")

def rag(user_query: str) -> str:
    """Executa a busca RAG com base na consulta do usuário."""
    logging.info(f"Recebendo pergunta: {user_query}")

    # Busca os documentos mais relevantes
    context = db.similarity_search_with_relevance_scores(user_query, k=3)
    relevant_context = [doc for doc, score in context if score >= 0.7]

    if not relevant_context:
        logging.warning("Nenhum contexto relevante encontrado para a pergunta.")
        return "Eu não sou capaz de responder a essa pergunta."

    formatted_context = "\n\n".join(
        [f"## Documento {i+1}\n{doc.page_content}\nSource: {doc.metadata.get('source', '')}"
         for i, doc in enumerate(relevant_context)]
    )

    chain = (template | 
             ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY) | 
             StrOutputParser())

    logging.debug(f"Enviando prompt ao ChatOpenAI:\n{formatted_context}")

    resposta = chain.invoke({"contexto": formatted_context, "pergunta": user_query})

    logging.debug(f"Resposta recebida do ChatOpenAI:\n{resposta}")

    return resposta

if __name__ == "__main__":
    print("Chat RAG - Pergunte sobre a história de Anwar!")
    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o chat. Até mais!")
            break
        resposta = rag(pergunta)
        print("\nResposta:\n", resposta)
