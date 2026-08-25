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