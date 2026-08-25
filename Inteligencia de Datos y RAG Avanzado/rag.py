## LangSmith y RAG con LangChain y Ollama
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.output_parsers import CommaSeparatedListOutputParser

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
# modelo.invoke(pregunta)
# python rag.py

# ollama pull
# ollama pull bge-m3:367m
# ollama pull bge-m3
# python rag.py
# Instalando paquetes necesarios y revisando resultados

# pip install faiss-cpu
# pip install -q langchain-community faiss-cpu

# Ilustrando el proceso con código
trechos = retriever.invoke(pregunta)
contexto = "\n\n".join(trecho.page_content for trecho in trechos)
cadena.invoke({"query": pregunta, "contexto":contexto})

# Integrando en la Rack y configurando RAGChain
from langchain_core.runnables import RunnablePassthrough


#rag_chain = (
#    {"contexto": RunnablePassthrough() | retriever,
#     "query": RunnablePassthrough()
#    }
#    | prompt
#    | modelo
#    | StrOutputParser()
#)

# Invocando RAGChain y próximos pasos
#rag_chain.invoke(pregunta)
query_model = OllamaLLM(model="gemma3:1b")

# Creando el template para reescribir preguntas

rewriter_prompt_template = """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta más precisa por medio de la búsqueda semántica.
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""

# Creando y ejecutando la cadena de reescritura

from langchain_core.prompts import PromptTemplate
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)

rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

# rewriter_chain.invoke(pregunta)

# Evaluando y ajustando la salida del modelo

# rag_chain = (
#    {
#        "contexto": RunnablePassthrough() | rewriter_chain | retriever,
#        "query":RunnablePassthrough()
#    }
#    | prompt | modelo | StrOutputParser()
#)

# Mejorando el modelo y evaluando resultados
#query_model = OllamaLLM(model="gemma3:4b")

template_multipregunta = """
Eres un asistente de modelo de lenguajes de IA. Tu tarea es generar cinco versiones diferentes de la pregunta 
del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar multiples
perspectivas sobre la pregunta del usuario, tu objetivo es auxiliar al usuario a superar algunas de las
limitaciones de la búsqueda por similitud basada en distancia. Debes generar únicamente las preguntas alternativas
separadas en filas diferentes (new line) sin ningún texto adicional.

# PREGUNTA ORIGINAL: {question}

# FORMATO DE SALIDA :
["primera pregunta","segunda pregunta",...,"quinta pregunta"]
"""

prompt_multipregunta = PromptTemplate.from_template(template_multipregunta)

chain_multipregunta = prompt_multipregunta | modelo | CommaSeparatedListOutputParser()

preguntas = chain_multipregunta.invoke(pregunta)

print(preguntas)

rag_chain = (
   {
       "contexto": RunnablePassthrough() | rewriter_chain | retriever,
        "query":RunnablePassthrough()
   }
    | prompt | modelo | StrOutputParser()
)

# Executando a RackChain para cada pergunta

for p in preguntas:
    rag_chain.invoke(p)