import os
import re
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

# ── CONFIG — update these paths if yours differ ───────────────────────────────
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
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


# ── 2. PDF loader with OCR fallback ──────────────────────────────────────────

def load_pdf(path: str) -> List[Document]:
    docs = []
    pdf  = fitz.open(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()

        # If the page has very little text, OCR it
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


# ── 3. Clean extracted text ───────────────────────────────────────────────────

def clean_docs(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
        doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
        doc.page_content = re.sub(r'(?m)^\s*(Page\s*)?\d+\s*$', '', doc.page_content)
        doc.page_content = doc.page_content.strip()
    return [doc for doc in docs if len(doc.page_content) > 50]


# ── 4. Load any file type ─────────────────────────────────────────────────────

def load_document(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}  (supported: .txt, .pdf, .docx)")


def load_folder(folder: str) -> List[Document]:
    all_docs  = []
    supported = {".txt", ".pdf", ".docx", ".doc"}
    for file in Path(folder).iterdir():
        if file.suffix.lower() in supported:
            print(f"  Loading: {file.name}")
            all_docs.extend(load_document(str(file)))
    return all_docs


# ── 5. Ask the user what to load ──────────────────────────────────────────────

print("\n=== Document Q&A (RAG) ===\n")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(CHROMA_DIR):
    print("Found existing vector database.")
    print("  1. Use existing database (fast)")
    print("  2. Load new document(s) and rebuild")
    db_choice = input("\nEnter 1 or 2: ").strip()
else:
    db_choice = "2"

if db_choice == "1":
    print("Loading existing database...")
    vectorstore    = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    chunks         = None
    bm25_available = False
else:
    bm25_available = True
    print("\nLoad a single file or an entire folder?")
    print("  1. Single file")
    print("  2. Entire folder")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        path = input("Enter file path: ").strip()
        docs = load_document(path)
    elif choice == "2":
        folder = input("Enter folder path: ").strip()
        docs   = load_folder(folder)
    else:
        print("Defaulting to acme_policy.txt")
        docs = load_document("acme_policy.txt")

    docs   = clean_docs(docs)
    print(f"\nLoaded {len(docs)} page(s) after cleaning")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and saving to disk...")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    print("Saved! Future runs will load instantly.")


# ── 6. Build retrievers ───────────────────────────────────────────────────────

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

if bm25_available and chunks:
    bm25_retriever      = BM25Retriever.from_documents(chunks)
    bm25_retriever.k    = 4
    retriever           = HybridRetriever(bm25=bm25_retriever, vector=vector_retriever)
    print("Using hybrid search (BM25 + vector).")
else:
    retriever = vector_retriever
    print("Using vector search.")


# ── 7. Prompt + chain ─────────────────────────────────────────────────────────

prompt = PromptTemplate.from_template("""
Answer the question using only the context below.
If the answer isn't in the context, say "I don't know."

Context: {context}

Question: {question}
""")

llm = OllamaLLM(model="llama3.2")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ── 8. Q&A loop ───────────────────────────────────────────────────────────────

print("\nReady! Ask anything about your document(s).")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.strip().lower() == "quit":
        break
    if not question.strip():
        continue
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")