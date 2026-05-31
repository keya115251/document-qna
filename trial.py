import os
import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
import pytesseract
from pdf2image import convert_from_path
import fitz  # pymupdf

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

# ── CONFIG ────────────────────────────────────────────────────────────────────
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH   = r"C:\poppler\poppler-26.02.0\Library\bin"
CHROMA_DIR     = "./chroma_db"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ── 1. Hybrid retriever ───────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    bm25: BM25Retriever
    vector: object = Field(default=None)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25_results   = self.bm25.invoke(query)
        vector_results = self.vector.invoke(query)
        seen, combined = set(), []
        for doc in bm25_results + vector_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[:6]


# ── 2. URL loader ─────────────────────────────────────────────────────────────

def load_url(url: str) -> List[Document]:
    print(f"  Fetching: {url}")
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if not text:
        raise ValueError("No text could be extracted from the URL.")
    return [Document(page_content=text, metadata={"source": url, "page": 1})]


# ── 3. PDF loader with OCR fallback ──────────────────────────────────────────

def load_pdf(path: str) -> List[Document]:
    docs = []
    pdf  = fitz.open(path)
    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()
        if len(text) < 50:
            print(f"    Page {page_num + 1}: no text found, running OCR...")
            images = convert_from_path(
                path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=POPPLER_PATH
            )
            if images:
                text = pytesseract.image_to_string(images[0])
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": page_num + 1}
            ))
    pdf.close()
    return docs


# ── 4. Clean extracted text ───────────────────────────────────────────────────

def clean_docs(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
        doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
        doc.page_content = re.sub(r'(?m)^\s*(Page\s*)?\d+\s*$', '', doc.page_content)
        doc.page_content = doc.page_content.strip()
    return [doc for doc in docs if len(doc.page_content) > 50]


# ── 5. Load any file type ─────────────────────────────────────────────────────

def load_document(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder: str) -> List[Document]:
    all_docs  = []
    supported = {".txt", ".pdf", ".docx", ".doc"}
    for file in Path(folder).iterdir():
        if file.suffix.lower() in supported:
            print(f"  Loading: {file.name}")
            all_docs.extend(load_document(str(file)))
    return all_docs


# ── 6. Build vector store from docs ──────────────────────────────────────────

def build_vectorstore(docs: List[Document], embeddings) -> tuple:
    docs     = clean_docs(docs)
    print(f"Loaded {len(docs)} page(s) after cleaning")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    print("Embedding and saving to disk...")
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Saved!")
    return vs, chunks


# ── 7. Initial load ───────────────────────────────────────────────────────────

print("\n=== Document Q&A (RAG) ===\n")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(CHROMA_DIR):
    print("Found existing vector database.")
    print("  1. Use existing database (fast)")
    print("  2. Load new document(s) and rebuild")
    db_choice = input("\nEnter 1 or 2: ").strip()
else:
    db_choice = "2"

all_chunks = []

if db_choice == "1":
    print("Loading existing database...")
    vectorstore    = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    bm25_available = False
else:
    bm25_available = True
    print("\nWhat would you like to load?")
    print("  1. Single file")
    print("  2. Entire folder")
    print("  3. URL")
    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        path = input("Enter file path: ").strip()
        docs = load_document(path)
    elif choice == "2":
        folder = input("Enter folder path: ").strip()
        docs   = load_folder(folder)
    elif choice == "3":
        url  = input("Enter URL: ").strip()
        docs = load_url(url)
    else:
        print("Defaulting to acme_policy.txt")
        docs = load_document("acme_policy.txt")

    vectorstore, all_chunks = build_vectorstore(docs, embeddings)


# ── 8. Build retrievers ───────────────────────────────────────────────────────

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

if bm25_available and all_chunks:
    bm25_retriever   = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 4
    retriever        = HybridRetriever(bm25=bm25_retriever, vector=vector_retriever)
    print("Using hybrid search (BM25 + vector).")
else:
    retriever = vector_retriever
    print("Using vector search.")


# ── 9. Prompts + LLM ─────────────────────────────────────────────────────────

llm      = OllamaLLM(model="llama3.2")
fast_llm = OllamaLLM(model="llama3.2:1b")

answer_prompt = PromptTemplate.from_template("""
You are a helpful assistant with memory of the conversation so far.
Answer the question using the context below and the conversation history.
If the answer isn't explicitly stated but can be reasonably inferred, answer it.
If there is truly no relevant information, say "I don't know."

Conversation history:
{history}

Context from documents:
{context}

Question: {question}

Answer:""")

rewrite_prompt = PromptTemplate.from_template("""Rewrite this question as a better search query. Return only the query, nothing else.

Question: {question}
History summary: {history}""")

summary_prompt = PromptTemplate.from_template("""
Summarize the following document content in 4-5 sentences.
Cover the main topics, key points, and what a reader would learn from it.

Content:
{content}

Summary:""")

rewrite_chain = rewrite_prompt | fast_llm | StrOutputParser()
answer_chain  = answer_prompt  | llm      | StrOutputParser()
summary_chain = summary_prompt | llm      | StrOutputParser()


# ── 10. Auto summary ─────────────────────────────────────────────────────────

def summarize_docs(docs: List[Document]) -> None:
    sample  = "\n\n".join(doc.page_content for doc in docs[:3])[:3000]
    print("\nGenerating document summary...")
    summary = summary_chain.invoke({"content": sample})
    print(f"\n📄 Summary:\n{summary}")
    print("-" * 60)


# ── 11. Chat history export ───────────────────────────────────────────────────

def export_history(history: list) -> None:
    if not history:
        print("No conversation to export.\n")
        return
    from datetime import datetime
    filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Document Q&A — Chat Export ===\n")
        f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for i, turn in enumerate(history, 1):
            f.write(f"Q{i}: {turn['question']}\n\n")
            f.write(f"A{i}: {turn['answer']}\n\n")
            if turn.get("citations"):
                f.write("Sources:\n")
                for c in turn["citations"]:
                    f.write(f"  • {c}\n")
            f.write("-" * 60 + "\n\n")
    print(f"✅ Chat exported to {filename}\n")


# ── 12. Format helpers ────────────────────────────────────────────────────────

def format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-4:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)

def format_docs_with_citations(docs: List[Document]) -> tuple[str, list]:
    context_parts, citations = [], []
    for doc in docs:
        source       = doc.metadata.get("source", "unknown")
        label_source = source if source.startswith("http") else Path(source).name
        page         = doc.metadata.get("page", "?")
        label        = f"[SOURCE: {label_source}, PAGE: {page}]"
        context_parts.append(f"{label}\n{doc.page_content}")
        citation = f"{label_source} — p.{page}"
        if citation not in citations:
            citations.append(citation)
    return "\n\n".join(context_parts), citations

def get_docs(question: str, history: list) -> List[Document]:
    last      = f"Previous topic: {history[-1]['question']}" if history else "None"
    rewritten = rewrite_chain.invoke({"question": question, "history": last})
    query     = rewritten.strip().split("\n")[0]
    queries   = [question, query] if query != question else [question]
    seen, all_docs = set(), []
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return all_docs[:8]


# ── 13. Run auto summary after loading ───────────────────────────────────────

if db_choice != "1":
    summarize_docs(docs)


# ── 14. Q&A loop ─────────────────────────────────────────────────────────────

print("\nReady! Ask anything about your documents.")
print("Commands:")
print("  'quit'              → exit")
print("  'export'            → save chat history to a file")
print("  'clear'             → reset conversation memory")
print("  'load url <url>'    → add a webpage mid-session\n")

history = []

while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() == "quit":
        break

    if question.lower() == "export":
        export_history(history)
        continue

    if question.lower() == "clear":
        history = []
        print("Conversation history cleared.\n")
        continue

    # ── Load a URL mid-session ────────────────────────────────────────────────
    if question.lower().startswith("load url "):
        url = question[9:].strip()
        try:
            new_docs   = load_url(url)
            new_docs   = clean_docs(new_docs)
            splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            new_chunks = splitter.split_documents(new_docs)
            vectorstore.add_documents(new_chunks)
            all_chunks.extend(new_chunks)
            bm25_retriever             = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k           = 4
            retriever.__dict__['bm25'] = bm25_retriever
            print(f"Added {len(new_chunks)} chunks from {url}")
            summarize_docs(new_docs)
        except Exception as e:
            print(f"Failed to load URL: {e}\n")
        continue

    # ── Normal Q&A ────────────────────────────────────────────────────────────
    docs               = get_docs(question, history)
    context, citations = format_docs_with_citations(docs)
    answer             = answer_chain.invoke({
        "context":  context,
        "question": question,
        "history":  format_history(history)
    })

    print(f"\nAnswer: {answer}")
    print("\nSources:")
    for c in citations:
        print(f"  • {c}")
    print()

    history.append({"question": question, "answer": answer, "citations": citations})