from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from vectorstore import build_vectorstore
from chunker import chunk_docs
from loader import load_pdf

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore, k: int = 3, model: str = "llama3.1"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=model)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
