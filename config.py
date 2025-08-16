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

# Pinecone client (v2-style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if it doesn't exist (1536 dims for text-embedding-3-small)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)

index = pinecone.Index(INDEX_NAME)