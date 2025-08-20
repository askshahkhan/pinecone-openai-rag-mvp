import streamlit as st
from config import Config
from embeddings import Embedder
from vectorstore import VectorStore
from query import QueryEngine

# -----------------------------
# Initialize components
# -----------------------------
st.set_page_config(page_title="RAG MVP", page_icon="📚")
st.title("Pinecone + OpenAI RAG MVP")

# Initialize config and embedder
config = Config()
embedder = Embedder()
store = VectorStore(config, embedder)
engine = QueryEngine(store)

# -----------------------------
# Tabs for Upsert / Query
# -----------------------------
tab1, tab2 = st.tabs(["Upsert Docs", "Ask Question"])

# --------- Upsert Docs ---------
with tab1:
    st.header("📝 Upsert Documents")
    docs_input = st.text_area(
        "Enter documents (one per line):",
        height=200
    )
    
    if st.button("Upsert Documents"):
        if not docs_input.strip():
            st.warning("Please enter at least one document.")
        else:
            docs = [line.strip() for line in docs_input.strip().split("\n") if line.strip()]
            store.add_documents(docs)
            st.success(f"✅ {len(docs)} documents upserted into Pinecone!")

# --------- Ask Question ---------
with tab2:
    st.header("❓ Ask a Question")
    question = st.text_input("Enter your question here:")
    
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = engine.ask(question)
            st.markdown(f"**Answer:** {answer}")