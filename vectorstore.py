from langchain_pinecone import PineconeVectorStore

class VectorStore:
    def __init__(self, config, embedder):
        self.config = config
        self.embedder = embedder
        self.store = PineconeVectorStore(
            index_name=config.INDEX_NAME,
            embedding=embedder.get()
        )

    def add_documents(self, docs):
        """Upsert a list of text documents into Pinecone."""
        self.store.add_texts(docs)
        print(f"✅ {len(docs)} documents upserted into Pinecone")

    def as_retriever(self, k: int = 3):
        """Return a retriever for querying."""
        return self.store.as_retriever(search_kwargs={"k": k})