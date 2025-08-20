from typing import List
from config import Config

# Purpose of this is to put words into numbers
# Turning sentences into vectors that represents their meaning
# List of numbers is called an embedding

class Embedder:
    EMBED_MODEL = "text-embedding-3-small"  # 1536-dim

    def __init__(self, config: Config):
        self.openai_client = config.openai_client

    def embed_text(self, text: str) -> List[float]:
        """Convert text into an embedding vector."""
        resp = self.openai_client.embeddings.create(
            input=text,
            model=self.EMBED_MODEL
        )
        return resp.data[0].embedding

# vec = embed_text("Hello world")
# It sends the words "Hello world" to OpenAI’s embedding model (text-embedding-3-small)
# OpenAI replies with 1,536 numbers (a vector)
# Each number captures a little bit of the “meaning” of the text
# So "Hello world" → [0.0021, -0.0153, 0.0098, ...] (1,536 numbers long)