import chromadb
from langchain_ollama import OllamaEmbeddings

# --- Step 1: Connect to Chroma and load collection ---
client = chromadb.PersistentClient(path="data/chroma_db/smollm3")
collection = client.get_collection(name="cpu_information_docs")

# --- Step 2: Initialize embeddings model ---
embeddings = OllamaEmbeddings(model="alibayram/smollm3:latest")

# --- Step 3: Prepare your query ---
query = "Which CPU is best for a server with high multi-thread performance?"
query_embedding = embeddings.embed_query(query)

# --- Step 4: Perform similarity search ---
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# --- Step 5: Inspect the results ---
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print("Document chunk:", doc)
    print("Metadata:", meta)