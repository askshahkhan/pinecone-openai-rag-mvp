import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# 1. Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("INDEX_NAME", "mystical-index")

# -----------------------------
# 2. Initialize Pinecone client
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,       # matches text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# -----------------------------
# 3. Embeddings + VectorStore
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# -----------------------------
# 4. Insert docs
# -----------------------------
docs = [
    "The village of Brossenwald holds an annual competition where locals carve instruments out of ice and play them until they melt.",
    "In the archives of the Marlowe Observatory, there is a log of an unexplained comet sighting from 1837 that appears in no modern astronomy catalog.",
    "The mineral 'amberite' was once believed to glow faintly under moonlight, though modern tests suggest the effect was an optical illusion.",
    "On the island of Verdanthold, farmers still use wind-powered seed scatterers that date back over 300 years.",
    "The Clock of Drelmere runs backwards during the solstice, a phenomenon that has baffled horologists for centuries.",
    "An old shipping ledger describes a cargo of 'whistling sand' that supposedly produced tones when poured from one vessel to another.",
    "During the Festival of Lantern Shadows in Hallowmere, children wear masks painted with patterns that only reveal meaning under candlelight.",
    "The lost manuscript of Ardin Telvash reportedly contains recipes for inks that never fade, even after centuries.",
    "Travelers in the Marshes of Yorm speak of a moss that, when stepped on, produces a faint metallic ringing sound.",
    "The town records of East Grindle mention a bridge built entirely of petrified wood, dismantled in 1892 after it was deemed too eerie."
]

vectorstore.add_texts(docs)

print(f"✅ {len(docs)} docs upserted to Pinecone index: {INDEX_NAME}")

# -----------------------------
# 5. Retrieval QA pipeline
# -----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# -----------------------------
# 6. Ask a question
# -----------------------------
question = "What is special about the Clock of Drelmere?"
answer = qa_chain.run(question)
print("Q:", question)
print("A:", answer)