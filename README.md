# Marginalia — Local RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDF documents and ask questions about their content. No API keys, no external calls — embeddings, vector storage, and generation all run on your own machine.

## How it works

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace embeddings
    → Chroma vector store → retriever → Ollama (llama3.1) → answer
```

1. **Load** — PDFs are parsed page-by-page with `PyPDFLoader`.
2. **Chunk** — Pages are split into overlapping chunks with `RecursiveCharacterTextSplitter`, so retrieval can return focused, relevant passages instead of whole pages.
3. **Embed & store** — Each chunk is embedded locally using `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent Chroma vector database.
4. **Retrieve** — A user's question is embedded the same way, and the top-k most similar chunks are pulled from the vector store.
5. **Generate** — Retrieved chunks + the question are passed to a local LLM via [Ollama](https://ollama.com) (`llama3.1`), which answers grounded in that context.

Multiple PDFs can be uploaded over time — each new upload is **added** to the existing vector store rather than replacing it, so questions can draw on everything uploaded so far.

## Project structure

```
RAG/
├── main.py            # FastAPI app — /upload and /ask endpoints, serves the frontend
├── loader.py           # PDF loading
├── chunker.py          # Text splitting
├── vectorstore.py       # Embeddings + Chroma (build / load / add)
├── rag_chain.py         # Retriever + prompt + LLM chain (LCEL)
├── static/
│   └── index.html      # Simple chat frontend
├── uploaded_pdfs/       # Uploaded PDFs land here (gitignored)
├── chroma_db/           # Persisted vector store (gitignored)
└── pyproject.toml
```

## Setup

**1. Install [Ollama](https://ollama.com/download) and pull a model:**
```bash
ollama pull llama3.1
```
Make sure Ollama is running in the background before starting the API.

**2. Install Python dependencies** (using [uv](https://docs.astral.sh/uv/)):
```bash
uv add langchain langchain-community pypdf langchain-text-splitters \
       langchain-huggingface sentence-transformers langchain-chroma \
       langchain-ollama fastapi uvicorn python-multipart
```

**3. Run the server:**
```bash
uvicorn main:app --reload
```

**4. Open the app:**
Visit `http://127.0.0.1:8000/` in your browser for the chat UI, or `http://127.0.0.1:8000/docs` for the interactive API docs (Swagger).

## API endpoints

| Endpoint  | Method | Description |
|-----------|--------|-------------|
| `/upload` | POST   | Upload a PDF. Saves it to disk and processes it (chunk + embed + add to vector store) in the background. Returns immediately with a status message. |
| `/ask`    | POST   | Ask a question (JSON body: `{"question": "..."}`). Returns the answer grounded in all uploaded documents so far. |
| `/`       | GET    | Serves the chat frontend. |

## Notes

- Answers are generated only from uploaded document content passed into the prompt — the model is instructed not to use outside knowledge, though this depends on the LLM following that instruction rather than a hard technical restriction.
- Uploaded PDFs and the vector store persist across restarts unless their folders are deleted.
- This is a personal/learning project — not hardened for concurrent multi-user production use (e.g. simultaneous uploads can race on the shared vector store).

## Possible next steps

- Source citations in answers (chunk metadata already tracks originating file/page)
- Endpoint to list or remove specific uploaded documents
- Similarity-score thresholding to reject off-topic questions more reliably
