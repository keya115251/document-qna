# Document Q&A (RAG)

A local RAG-based Document Q&A system. Upload any document or paste a URL — ask questions in plain English and get accurate, cited answers. Runs fully offline with no API costs.

## Features

- **Hybrid search** — combines BM25 keyword search and vector search for better retrieval
- **Multi-document support** — load and query across an entire folder of documents at once
- **URL support** — paste any webpage URL and ask questions about it instantly
- **OCR** — extracts text from image-based and scanned PDFs automatically
- **Conversation memory** — ask follow-up questions naturally, the system remembers context
- **Citations** — every answer shows exactly which file and page it came from
- **Query rewriting** — vague questions are automatically rewritten for better retrieval
- **Persistent storage** — vector DB saves to disk, reloads instantly on future runs

## Stack

- **LangChain** — pipeline orchestration
- **Ollama** — local LLM (llama3.2) and embeddings (nomic-embed-text), runs offline
- **Chroma** — local vector database
- **Tesseract** — OCR engine for image-based PDFs
- **BeautifulSoup** — web scraping for URL support

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
```bash
ollama pull llama3.2
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

2. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (Windows)

3. Clone the repo and create a virtual environment:
```bash
git clone https://github.com/keya115251/document-qna.git
cd document-qna
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Update the paths in `trial.py` to match your Tesseract and Poppler install locations:
```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH   = r"C:\poppler\poppler-26.02.0\Library\bin"
```

6. Run:
```bash
python trial.py
```

## Usage

At startup, choose to load a single file, folder, or URL. During a session:

- Ask any question about your loaded documents
- Type `load url <url>` to add a webpage to the knowledge base mid-session
- Type `clear` to reset conversation memory
- Type `quit` to exit

## Roadmap

- [x] Hybrid search (BM25 + vector)
- [x] Multi-document support
- [x] OCR for image-based PDFs
- [x] Query rewriting
- [x] Citations with source and page attribution
- [x] Conversation memory
- [x] URL support with mid-session loading
- [x] Auto document summary on load
- [x] Chat history export
- [x] Web UI (Streamlit)
- [ ] Evaluation dashboard
