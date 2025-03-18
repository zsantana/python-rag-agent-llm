import os
import json
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Carregar texto de um arquivo externo
def carregar_texto(arquivo: str) -> str:
    with open(arquivo, "r", encoding="utf-8") as f:
        return f.read()

texto = carregar_texto("texto.txt")

documents = [Document(page_content=texto, metadata={"autor": "Anwar", "source": "https://medium.com/@anwarhermuche/apenas-tr%C3%AAs-quest%C3%B5es-eb2cce38a6f6"})]

# Dividir texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Criar base vetorial
db = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    persist_directory="db",
)

# Template do prompt
template = PromptTemplate.from_template("""
Answer the user query based on context. If you don’t know the answer or the context does not have the answer, say that you don't know.
ALWAYS answer in pt-BR. You are going to answer questions about Anwar's life. Suppose you're him. Answer the questions with "Eu fiz isso...", "Olha, eu sou um cara..."
ALWAYS answer with the source of your knowledge. I want the answer with a link. A URL.

## CONTEXT
{contexto}

## USER QUERY
{pergunta}
""")

def log_request_response(model, messages, response):
    """Loga a requisição e resposta da OpenAI."""
    log_data = {
        "request": {
            "model": model,
            "messages": messages
        },
        "response": response
    }
    logger.info(f"OpenAI Request-Response:\n{json.dumps(log_data, indent=2, ensure_ascii=False)}")

def rag(user_query: str) -> str:
    context = db.similarity_search_with_relevance_scores(user_query, k=3)
    context = list(filter(lambda x: x[1] >= 0.7, context))

    if not context:
        return "Eu não sou capaz de responder a essa pergunta."

    context_text = "\n\n".join([
        f"## Documento {k}\n{doc[0].page_content}\nSource: {doc[0].metadata.get('source', '')}"
        for k, doc in enumerate(context, start=1)
    ])

    messages = [
        {"role": "system", "content": "Responda em pt-BR sobre a vida de Anwar com base no contexto fornecido."},
        {"role": "user", "content": f"## CONTEXT\n{context_text}\n\n## USER QUERY\n{user_query}"}
    ]

    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Log da requisição
    logger.info(f"🔹 Enviando requisição para OpenAI:\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
    
    response = chat_model.invoke({"contexto": context_text, "pergunta": user_query})

    # Log da resposta
    logger.info(f"🔹 Resposta da OpenAI:\n{response}")

    return response

# Teste inicial
if __name__ == "__main__":
    pergunta = "como você começou em ciência de dados?"
    print(rag(pergunta))
