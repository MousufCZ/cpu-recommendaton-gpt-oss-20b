import logging
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
import torch
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
        logging.FileHandler(log_file),
        logging.StreamHandler()   
    ]
)
logging.info("Starting RAG inference/testing pipeline...")

# --- Step 1: Load existing ChromaDB and embeddings ---
chroma_path = "data/chroma_db/smollm3"  
collection_name = "cpu_docs_smollm3_ollama" 

logging.info(f"Loading ChromaDB from {chroma_path}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    num_docs = vectorstore._collection.count()
    logging.info(f"Loaded {num_docs} documents from ChromaDB")
    
except Exception as e:
    logging.error(f"Failed to load ChromaDB: {e}")
    logging.info("Run rag_pipeline.py first to create the vectorstore!")
    exit(1)

# --- Step 2: Setup Ollama SmolLM3 for Generation ---
logging.info("Initializing Ollama SmolLM3 for text generation...")
llm = ChatOllama(
    model="alibayram/smollm3:latest", 
    temperature=0.1, 
    base_url="http://localhost:11434"  
)

try:
    test_response = llm.invoke("Say 'Ollama is working!'")
    logging.info(f"Ollama SmolLM3 test: {test_response.content}")
except Exception as e:
    logging.error(f"Ollama connection failed: {e}")
    logging.info("Make sure 'ollama serve' is running and model is pulled")
    exit(1)

# --- Step 3: Create RAG Chain (The Magic!) ---
logging.info("Building RAG chain: HF embeddings → ChromaDB → Ollama generation...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True 
)

# --- Step 4: Test the Full Pipeline ---
logging.info("Testing full RAG pipeline...")
test_queries = [
    "Compare the CPU_mark and give me the top 3 names. Ensure the CPU is less than 3 years.",
    "What are the key factors for choosing a server CPU?"
]

for i, query in enumerate(test_queries, 1):
    logging.info(f"\nTest Query {i}: {query}")
    
    try:
        result = qa_chain.invoke({"query": query})
        
        print(f"\nSmolLM3 Response:")
        print("-" * 50)
        print(result["result"])
        print("-" * 50)
        print(f"\nRetrieved {len(result['source_documents'])} documents:")
        
        for j, doc in enumerate(result["source_documents"]):
            print(f"  {j+1}. {doc.page_content[:150]}...")
            print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
            print()
            
    except Exception as e:
        logging.error(f"Error with query '{query}': {e}")

logging.info("RAG inference pipeline completed successfully!")
logging.info(f"Loaded from ChromaDB: {chroma_path}")