import chromadb

client = chromadb.PersistentClient(path="data/chroma_db/smollm3")
collections = client.list_collections()
print([c.name for c in collections])