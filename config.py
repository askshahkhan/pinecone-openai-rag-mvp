import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric

class Config:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
        self.INDEX_NAME = os.getenv("INDEX_NAME")
        self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        print("✅ Pinecone client initialized")
        if not self.pc.has_index(name=self.INDEX_NAME):
            self.pc.create_index(
                name=self.INDEX_NAME,
                metric=Metric.COSINE,
                dimension=1536,
                spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            )
        print("Using index:", self.INDEX_NAME)
