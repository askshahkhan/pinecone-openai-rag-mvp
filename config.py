import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

class Config:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
        self.INDEX_NAME = os.getenv("INDEX_NAME", "mystical-index")

        # Init Pinecone client
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)

        # Create index if needed
        if self.INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.INDEX_NAME,
                dimension=1536,  # for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        print(f"✅ Using Pinecone index: {self.INDEX_NAME}")