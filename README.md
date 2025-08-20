# Pinecone + OpenAI RAG MVP

This project is a minimal **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **OpenAI**, and **Pinecone**.  
It lets you embed documents, store them in a vector database, and query them with natural language questions.

---

## Features
- **Configurable** OpenAI + Pinecone setup via `.env`
- **Embeddings** handled with `text-embedding-3-small`
- **Vector store** powered by Pinecone
- **Retriever + LLM** combined with LangChain’s `RetrievalQA`
- **Modular design** (separate classes for config, embeddings, vectorstore, and query)

---

## Project Structure

```
pinecone-openai-rag-mvp/
│
├── config.py        # Pinecone + environment config
├── embeddings.py    # OpenAI embeddings wrapper
├── vectorstore.py   # Pinecone vector store wrapper
├── query.py         # RetrievalQA query engine
├── main.py          # Example usage (docs ingestion + QA)
├── .gitignore   
├── .env
├── requirements.py 
└── README.md
```