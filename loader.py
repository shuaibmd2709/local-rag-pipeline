from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path:str):
    loaders = PyPDFLoader(path)
    return loaders.load()
