from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb

# --- Step 1: Load documents ---
pdf_loader = PyPDFLoader("data/documents/cpu_guide.pdf")
csv_loader = CSVLoader("data/final_dataset/cpu_benchmarks_v5_final.csv")

docs = pdf_loader.load() + csv_loader.load()

# --- Step 2: Split documents into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs_split = splitter.split_documents(docs)

# --- Step 3: Create embeddings ---
embeddings = OpenAIEmbeddings()  # or HuggingFaceEmbeddings()

# --- Step 4: Store in Chroma ---
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="cpu_docs")

for i, doc in enumerate(docs_split):
    collection.add(
        ids=[str(i)],
        documents=[doc.page_content],
        metadatas=[doc.metadata]
    )