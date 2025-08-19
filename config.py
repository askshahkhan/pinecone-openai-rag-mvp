# config.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("INDEX_NAME")

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("✅ OpenAI client initialized")

# Pinecone client (v2-style)
pc = Pinecone(api_key=PINECONE_API_KEY) 

print("✅ Pinecone client initialized")

if not pc.has_index(name=INDEX_NAME):
    pc.create_index(
    name=INDEX_NAME,
    metric=Metric.COSINE,
    dimension=1536,
    spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
)

print("Using index:", INDEX_NAME)
