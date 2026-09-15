import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from loader import load_pdf
from chunker import chunk_docs
from vectorstore import build_vectorstore, load_vectorstore, add_to_vectorstore
from rag_chain import build_rag_chain

UPLOAD_DIR = "./uploaded_pdfs"
PERSIST_DIR = "./chroma_db"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


os.makedirs(UPLOAD_DIR, exist_ok=True)
vectorstore = None
rag_chain = None

if os.path.exists(PERSIST_DIR):
    print("Existing vector store found — loading from disk")
    vectorstore = load_vectorstore(PERSIST_DIR)
    rag_chain = build_rag_chain(vectorstore)
else:
    print("No vector store found yet — waiting for a PDF upload")


class Query(BaseModel):
    question: str


def process_pdf(file_path: str):
    """Runs in the background — loads, chunks, embeds, and adds the PDF to the vectorstore."""
    global vectorstore, rag_chain

    docs = load_pdf(file_path)
    chunks = chunk_docs(docs)

    if vectorstore is None:
        vectorstore = build_vectorstore(chunks, PERSIST_DIR)
    else:
        vectorstore = add_to_vectorstore(vectorstore, chunks)

    rag_chain = build_rag_chain(vectorstore)
    print(f"Finished processing {file_path} — {vectorstore._collection.count()} total chunks in store")


@app.post("/upload")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(process_pdf, file_path)

    return {"filename": file.filename, "status": "upload received, processing in background"}


@app.post("/ask")
def ask(query: Query):
    if rag_chain is None:
        return {"error": "No PDF has been fully processed yet. Try again shortly, or upload one via /upload first."}
    answer = rag_chain.invoke(query.question)
    return {"question": query.question, "answer": answer}