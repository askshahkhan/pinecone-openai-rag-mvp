from dotenv import load_dotenv
import os
from config import pc, INDEX_NAME
from embeddings import embed_text

load_dotenv()

# Example text to embed
text = "This is an example document about Pinecone and OpenAI."

# Generate embedding for the text
embedding = embed_text(text)

# Connect to the index
index = pc.Index(INDEX_NAME)

# Upsert into Pinecone
index.upsert(vectors=[
    {
        "id": "doc-1",         # unique identifier for the vector
        "values": embedding,   # the actual vector
        "metadata": {"text": text}
    }
])

print("✅ Document upserted into Pinecone")