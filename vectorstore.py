from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chunker import chunk_docs
from loader import load_pdf


embed_model = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore(chunks,persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name = embed_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

def load_vectorstore(persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name = embed_model)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
def add_to_vectorstore(vectorstore,chunks):
    vectorstore.add_documents(chunks)
    return vectorstore
