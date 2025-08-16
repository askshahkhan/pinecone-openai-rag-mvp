# config.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import pinecone

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.getenv("INDEX_NAME")

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("✅ OpenAI client initialized")

# Pinecone client (v2-style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

print("✅ Pinecone client initialized")

index = pinecone.Index(INDEX_NAME)

print("Using index:", INDEX_NAME)