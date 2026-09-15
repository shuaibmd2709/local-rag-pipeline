from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdf

def chunk_docs(documents,chunk_size:int = 1000, chunk_overlap:int = 100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
