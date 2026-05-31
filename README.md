# Document Q&A (RAG)

A local RAG-based document Q&A system built with LangChain and Ollama. Ask questions about your documents — runs fully offline, no API costs.

## Stack
- **LangChain** — pipeline orchestration
- **Ollama** — local LLM (llama3.2) and embeddings (nomic-embed-text)
- **Chroma** — local vector database

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Clone the repo and create a virtual environment:
```bash
git clone https://github.com/keya115251/document-qna.git
cd document-qna
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run:
```bash
python trial.py
```

## Usage
Place your document in the project folder, update the filename in `trial.py`, and run. Supports `.txt`, `.pdf`, and `.docx` files.

## Roadmap

- [ ] **Hybrid search** — combine BM25 keyword search with vector search for better retrieval accuracy
- [ ] **Multi-document support** — load and query across an entire folder of documents at once
- [ ] **Citations** — show exactly which part of the source document each answer came from
- [ ] **Web UI** — browser-based interface using Streamlit or Gradio so anyone can use it without the terminal