import logging
import tiktoken
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import retrieval_qa
import chromadb
import os
import shutil
import torch

# --- Setup logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "cpu_rag_pipeline.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # save to file
        logging.StreamHandler()         # still print to console
    ]
)
logging.info("Starting mixed RAG pipeline: Hugging Face embeddings + Ollama generation...")

# --- Step 1: Load documents ---
pdf_path = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/documents/understanding_server_cpus_and_how_to_choose_server_cpu.pdf"
csv_path = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/documents/cpu_text_chunks_2025-09-10_16-38-35.csv"

logging.info("Loading PDF and CSV documents...")
pdf_loader = PyPDFLoader(pdf_path)
csv_loader = CSVLoader(csv_path)
docs = pdf_loader.load() + csv_loader.load()
logging.info(f"Loaded {len(docs)} documents.")

# --- Step 2: Split documents into chunks ---
chunk_size = 800
chunk_overlap = 200
logging.info(f"Splitting documents into chunks of {chunk_size} characters with {chunk_overlap} overlap...")
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs_split = splitter.split_documents(docs)
logging.info(f"Created {len(docs_split)} text chunks.")

# --- Step 3: Setup Hugging Face Embeddings (384 dimensions) ---
logging.info("Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# Test embedding dimensions
test_embedding = embeddings.embed_query("Test CPU query")
logging.info(f"Hugging Face embedding dimension: {len(test_embedding)} (expected: 384)")


# --- Step 4: Setup ChromaDB with 384-dim embeddings ---
chroma_path = "data/chroma_db/smollm3"
try:
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        logging.info(f"Cleared old ChromaDB at {chroma_path}")
except Exception as e:
    logging.warning(f"Could not clear old ChromaDB: {e}")

os.makedirs(chroma_path, exist_ok=True)

logging.info("Creating ChromaDB collection with Hugging Face embeddings...")
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    collection_name="cpu_docs_smollm3_ollama",
    persist_directory=chroma_path
)

logging.info(f"ChromaDB created with {len(docs_split)} documents (384-dim vectors)")