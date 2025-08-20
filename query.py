from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

class QueryEngine:
    def __init__(self, vectorstore, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )

    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        return self.qa_chain.run(question)