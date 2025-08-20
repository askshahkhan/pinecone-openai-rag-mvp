from config import Config
from embeddings import Embedder

class QueryEngine:
    def __init__(self, config: Config, embedder: Embedder):
        self.config = config
        self.embedder = embedder
        self.index = config.pc.Index(config.INDEX_NAME)
        self.openai_client = config.openai_client

    def ask_question(self, question: str, top_k: int = 3) -> str:
        query_embedding = self.embedder.embed_text(question)
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = (
            f"Answer the following question using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    config = Config()
    embedder = Embedder(config)
    engine = QueryEngine(config, embedder)
    question = "What is special about the Clock of Drelmere?"
    answer = engine.ask_question(question)
    print(answer)