from langchain_openai import OpenAIEmbeddings

class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embedder = OpenAIEmbeddings(model=model)

    def get(self):
        """Return the LangChain embeddings object."""
        return self.embedder