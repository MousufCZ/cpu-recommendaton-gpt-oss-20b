import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
collections = client.list_collections()
print([c.name for c in collections])