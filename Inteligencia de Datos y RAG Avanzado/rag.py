import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

pdfs = DirectoryLoader('documentos', glob='*.pdf').load()
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, chunk_size=1250, chunk_overlap=150
)

fragmentos = splitter.split_documents(pdfs)

embeddings = OllamaEmbeddings(model='bge-m3:367m')

vector_store = FAISS.from_documents(documents=fragmentos, embedding=embeddings)

### Configurando el ChatPromptTemplate y el Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [("system", "Responde usando exclusivamente el contenido que se incluye a continuación. Genera una "),
     ("human", "{query}")]
)

retriever = vector_store.as_retriever()
modelo = OllamaLLM(model = "gemma3:4b")
cadena = prompt | modelo | StrOutputParser()

### Ejecutando consultas y solucionando errores

pregunta = 'Cómo solicitar el seguro de viaje?'
modelo.invoke(pregunta)
# python rag.py

# ollama pull
# ollama pull bge-m3:367m
# ollama pull bge-m3
# python rag.py
# Instalando paquetes necesarios y revisando resultados

# pip install faiss-cpu
# pip install -q langchain-community faiss-cpu
