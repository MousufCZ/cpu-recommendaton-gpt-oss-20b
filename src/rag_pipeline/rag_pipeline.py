import logging
import tiktoken
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import os

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
logging.info("Starting document ingestion pipeline...")

# --- Step 1: Load documents ---
pdf_path = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/data/documents/understanding_server_cpus_and_how_to_choose_server_cpu.docx"
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

# --- Step 3: Count tokens ---
encoder = tiktoken.get_encoding("gpt2")
for i, doc in enumerate(docs_split):
    num_tokens = len(encoder.encode(doc.page_content))
    logging.info(f"Chunk {i} uses {num_tokens} tokens.")

# --- Step 4: Create embeddings ---
logging.info("Creating embeddings...")
embeddings = OllamaEmbeddings(model="gpt-oss:20b")

# --- Step 5: Store in Chroma ---
chroma_path = "data/chroma_db"
os.makedirs(chroma_path, exist_ok=True)
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(name="cpu_information_docs")

logging.info("Adding chunks to Chroma database...")
for i, doc in enumerate(docs_split):
    collection.add(
        ids=[str(i)],
        documents=[doc.page_content],
        metadatas=[doc.metadata]
    )
    if (i+1) % 10 == 0:
        logging.info(f"Added {i+1}/{len(docs_split)} chunks to Chroma.")

logging.info("Pipeline completed successfully!")