import os
import re
import shutil
import time
from pathlib import Path
from typing import List
from datetime import datetime

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytesseract
from pdf2image import convert_from_path
import fitz
from fpdf import FPDF

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

st.set_page_config(page_title="Librarium", page_icon="📖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --parchment:   #ede8dc;
    --warm-white:  #f7f3ec;
    --ink:         #2c2416;
    --ink-light:   #5c4f3a;
    --sepia:       #8b6f47;
    --sepia-light: #c4a882;
    --gold:        #9a7000;
    --border:      #c8b898;
}
html, body, [class*="css"] { font-family: 'EB Garamond', Georgia, serif; background-color: var(--warm-white); color: var(--ink); }
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

.librarium-title { font-family: 'Playfair Display', Georgia, serif; font-size: 2.8rem; font-weight: 600; color: var(--sepia); letter-spacing: 0.02em; margin-bottom: 0; line-height: 1.1; }
.librarium-subtitle { font-family: 'EB Garamond', Georgia, serif; font-style: italic; color: var(--sepia); font-size: 1.05rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
.title-divider { border: none; border-top: 1.5px solid var(--border); margin: 0.5rem 0 1.5rem; }

section[data-testid="stSidebar"] { background-color: var(--parchment); border-right: 1.5px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-heading { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: var(--ink); font-weight: 600; margin-bottom: 0.3rem; }
.sidebar-sub { font-size: 0.82rem; color: var(--sepia); font-style: italic; margin-bottom: 1rem; }

.stButton > button { font-family: 'EB Garamond', serif; background-color: var(--ink); color: var(--parchment); border: 2px solid var(--ink); border-radius: 3px; padding: 0.45rem 1.2rem; font-size: 0.95rem; letter-spacing: 0.04em; transition: all 0.2s; width: 100%; }
.stButton > button:hover { background-color: transparent; color: var(--ink); }

.stTextInput > div > div > input { font-family: 'EB Garamond', serif; font-size: 1rem; background-color: var(--warm-white); border: 1.5px solid var(--border); border-radius: 3px; color: var(--ink); }

.chat-user { background: var(--parchment); border-left: 3px solid var(--gold); border-radius: 0 6px 6px 0; padding: 0.8rem 1.1rem; margin: 0.8rem 0 0.4rem; font-size: 1rem; color: var(--ink); }
.chat-assistant { background: var(--warm-white); border: 1.5px solid var(--border); border-radius: 6px; padding: 1rem 1.2rem; margin: 0.4rem 0 0.2rem; font-size: 1rem; line-height: 1.7; color: var(--ink); }
.chat-label { font-family: 'Playfair Display', serif; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--sepia-light); margin-bottom: 0.3rem; }
.citations { margin-top: 0.7rem; padding-top: 0.6rem; border-top: 1px solid var(--border); font-size: 0.85rem; color: var(--sepia); font-style: italic; }
.citation-item::before { content: "§ "; color: var(--gold); }

.summary-box { background: var(--parchment); border: 1.5px solid var(--border); border-left: 4px solid var(--gold); border-radius: 0 6px 6px 0; padding: 1rem 1.2rem; margin: 1rem 0; font-style: italic; color: var(--ink-light); line-height: 1.7; }
.summary-label { font-family: 'Playfair Display', serif; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gold); margin-bottom: 0.4rem; font-style: normal; }

.progress-step { display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0; font-size: 0.9rem; }
.step-done { color: #3a6b3a; } .step-active { color: var(--gold); font-weight: 500; } .step-pending { color: var(--sepia-light); }

.badge { display: inline-block; font-size: 0.78rem; padding: 0.15rem 0.6rem; border-radius: 20px; font-family: 'EB Garamond', serif; margin-bottom: 0.3rem; }
.badge-green { background: #f5f0e0; color: #9a7000; border: 1px solid #c49a2a; }

/* Eval panel */
.eval-panel { background: var(--parchment); border: 1.5px solid var(--border); border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }
.eval-title { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: var(--ink); margin-bottom: 0.8rem; }
.eval-row { display: flex; justify-content: space-between; align-items: center; padding: 0.3rem 0; font-size: 0.88rem; color: #ede8dc; border-bottom: 1px dashed var(--border); }
.eval-row:last-child { border-bottom: none; }
.eval-val { font-weight: 500; color: #ffffff; }
.eval-bar-wrap { background: #e8e0d0; border-radius: 99px; height: 6px; width: 80px; display: inline-block; vertical-align: middle; margin-left: 6px; }
.eval-bar-fill { background: var(--gold); border-radius: 99px; height: 6px; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.5rem; }
.stat-box { background: var(--warm-white); border: 1px solid var(--border); border-radius: 6px; padding: 0.6rem 0.8rem; text-align: center; }
.stat-num { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: var(--gold); font-weight: 600; }
.stat-lbl { font-size: 0.75rem; color: var(--sepia); font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in {
    "history": [], "vectorstore": None, "all_chunks": [],
    "retriever": None, "summary": None, "sources_loaded": [],
    "eval_log": [], "show_eval": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── HYBRID RETRIEVER ──────────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    bm25: BM25Retriever
    vector: object = Field(default=None)
    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25_results   = self.bm25.invoke(query)
        vector_results = self.vector.invoke(query)
        seen, combined = set(), []
        for doc in bm25_results + vector_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content); combined.append(doc)
        return combined[:6]


# ── LOADERS ───────────────────────────────────────────────────────────────────
def load_url(url):
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside"]): tag.decompose()
    text = re.sub(r'\n{3,}', '\n\n', soup.get_text(separator="\n")).strip()
    if not text: raise ValueError("No text extracted.")
    return [Document(page_content=text, metadata={"source": url, "page": 1})]

def load_pdf(path):
    docs = []
    pdf  = fitz.open(path)
    for i, page in enumerate(pdf):
        text = page.get_text().strip()
        if len(text) < 50:
            imgs = convert_from_path(path, first_page=i+1, last_page=i+1, poppler_path=POPPLER_PATH)
            if imgs: text = pytesseract.image_to_string(imgs[0])
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path, "page": i+1}))
    pdf.close()
    return docs

def load_file(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":    return load_pdf(path)
    elif ext == ".docx": return Docx2txtLoader(path).load()
    elif ext == ".txt":  return TextLoader(path).load()
    else: raise ValueError(f"Unsupported: {ext}")

def clean_docs(docs):
    for doc in docs:
        doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
        doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
        doc.page_content = re.sub(r'(?m)^\s*(Page\s*)?\d+\s*$', '', doc.page_content)
        doc.page_content = doc.page_content.strip()
    return [d for d in docs if len(d.page_content) > 50]


# ── RETRIEVER BUILD ───────────────────────────────────────────────────────────
def build_retriever(chunks):
    emb = OllamaEmbeddings(model="nomic-embed-text")
    vs  = Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR)
    vr  = vs.as_retriever(search_kwargs={"k": 4})
    b   = BM25Retriever.from_documents(chunks); b.k = 4
    st.session_state.vectorstore = vs
    st.session_state.retriever   = HybridRetriever(bm25=b, vector=vr)
    st.session_state.all_chunks  = chunks

def add_to_retriever(new_chunks):
    st.session_state.vectorstore.add_documents(new_chunks)
    st.session_state.all_chunks.extend(new_chunks)
    ac = st.session_state.all_chunks
    b  = BM25Retriever.from_documents(ac); b.k = 4
    vr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    st.session_state.retriever = HybridRetriever(bm25=b, vector=vr)


# ── LLM + CHAINS ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_llms():
    return OllamaLLM(model="llama3.2"), OllamaLLM(model="llama3.2:1b")

def get_chains():
    llm, fast_llm = get_llms()
    ap = PromptTemplate.from_template("""You are a helpful assistant with memory of the conversation so far.
Answer the question using the context below and the conversation history.
If the answer isn't explicitly stated but can be reasonably inferred, answer it.
If there is truly no relevant information, say "I don't know."

Conversation history:
{history}

Context from documents:
{context}

Question: {question}

Answer:""")
    rp = PromptTemplate.from_template("""Rewrite this question as a better search query. Return only the query, nothing else.

Question: {question}
History summary: {history}""")
    sp = PromptTemplate.from_template("""Summarize the following document content in 4-5 sentences.
Cover the main topics, key points, and what a reader would learn from it.

Content:
{content}

Summary:""")
    return ap|llm|StrOutputParser(), rp|fast_llm|StrOutputParser(), sp|llm|StrOutputParser()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def format_history(history):
    if not history: return "No previous conversation."
    lines = []
    for t in history[-4:]:
        lines.append(f"User: {t['question']}")
        lines.append(f"Assistant: {t['answer']}")
    return "\n".join(lines)

def format_docs_with_citations(docs):
    parts, citations = [], []
    for doc in docs:
        src  = doc.metadata.get("source","unknown")
        lsrc = src if src.startswith("http") else Path(src).name
        page = doc.metadata.get("page","?")
        parts.append(f"[SOURCE: {lsrc}, PAGE: {page}]\n{doc.page_content}")
        c = f"{lsrc} - p.{page}"
        if c not in citations: citations.append(c)
    return "\n\n".join(parts), citations

def get_docs(question, history, rewrite_chain):
    last      = f"Previous topic: {history[-1]['question']}" if history else "None"
    rewritten = rewrite_chain.invoke({"question": question, "history": last})
    query     = rewritten.strip().split("\n")[0]
    queries   = [question, query] if query != question else [question]
    seen, all_docs = set(), []
    for q in queries:
        for doc in st.session_state.retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content); all_docs.append(doc)
    return all_docs[:8]

def compute_grounding_score(answer: str, docs: List[Document]) -> float:
    """What fraction of answer sentences are supported by retrieved context."""
    context  = " ".join(d.page_content.lower() for d in docs)
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
    if not sentences: return 0.0
    matched = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 4]
        if not words: continue
        hits = sum(1 for w in words if w in context)
        if hits / len(words) > 0.4:
            matched += 1
    return round(matched / len(sentences), 2)

def sanitize(text):
    replacements = {"\u2014":"-","\u2013":"-","\u2019":"'","\u2018":"'","\u201c":'"',"\u201d":'"',"\u2022":"*","\u00a0":" ","\u2026":"..."}
    for ch, r in replacements.items(): text = text.replace(ch, r)
    return text.encode("latin-1","replace").decode("latin-1")

def export_chat_pdf(history):
    pdf = FPDF()
    pdf.add_page(); pdf.set_margins(20,20,20); pdf.set_auto_page_break(True, margin=20)
    pdf.set_font("Helvetica","B",18); pdf.set_text_color(44,36,22)
    pdf.cell(0,12,"Librarium - Chat Export",ln=True)
    pdf.set_font("Helvetica","",10); pdf.set_text_color(120,100,70)
    pdf.cell(0,8,f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",ln=True)
    pdf.set_draw_color(200,184,152); pdf.line(20,pdf.get_y(),190,pdf.get_y()); pdf.ln(6)
    for i, t in enumerate(history, 1):
        pdf.set_font("Helvetica","B",11); pdf.set_text_color(44,36,22)
        pdf.multi_cell(0,7,sanitize(f"Q{i}: {t['question']}")); pdf.ln(2)
        pdf.set_font("Helvetica","",10); pdf.set_text_color(60,50,35)
        pdf.multi_cell(0,6,sanitize(t["answer"])); pdf.ln(2)
        if t.get("citations"):
            pdf.set_font("Helvetica","I",9); pdf.set_text_color(139,111,71)
            for c in t["citations"]: pdf.cell(0,5,sanitize(f"  * {c}"),ln=True)
        pdf.ln(3); pdf.set_draw_color(200,184,152); pdf.line(20,pdf.get_y(),190,pdf.get_y()); pdf.ln(5)
    return bytes(pdf.output())


# ── PROGRESS ─────────────────────────────────────────────────────────────────
def show_steps(area, steps):
    html = ""
    for label, state in steps:
        if state   == "done":   html += f'<div class="progress-step step-done">&#10003; {label}</div>'
        elif state == "active": html += f'<div class="progress-step step-active">&#8635; {label}...</div>'
        else:                   html += f'<div class="progress-step step-pending">&#9675; {label}</div>'
    area.markdown(html, unsafe_allow_html=True)

def run_indexing(docs, source_names):
    pa = st.empty()
    steps = [("Cleaning text","active"),("Chunking","pending"),("Embedding","pending"),("Generating summary","pending")]
    show_steps(pa, steps); docs = clean_docs(docs)
    steps[0]=("Cleaning text","done"); steps[1]=("Chunking","active"); show_steps(pa,steps)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    steps[1]=("Chunking","done"); steps[2]=("Embedding","active"); show_steps(pa,steps)
    if st.session_state.retriever is None:
        if os.path.exists(CHROMA_DIR): shutil.rmtree(CHROMA_DIR)
        build_retriever(chunks)
    else:
        add_to_retriever(chunks)
    steps[2]=("Embedding","done"); steps[3]=("Generating summary","active"); show_steps(pa,steps)
    _, _, sc = get_chains()
    sample   = "\n\n".join(d.page_content for d in docs[:3])[:3000]
    st.session_state.summary = sc.invoke({"content": sample})
    steps[3]=("Generating summary","done"); show_steps(pa,steps)
    for name in source_names:
        if name not in st.session_state.sources_loaded:
            st.session_state.sources_loaded.append(name)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-heading">📚 Librarium</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Your personal document library</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.sources_loaded:
        st.markdown('<p style="color: #2c2416; font-family: Playfair Display, serif; font-weight: 600;">Loaded sources:</p>', unsafe_allow_html=True)
        for src in st.session_state.sources_loaded:
            st.markdown(f'<span class="badge badge-green">&#10003; {src}</span>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown('<div class="sidebar-heading" style="font-size:0.95rem">Load Documents</div>', unsafe_allow_html=True)
    load_mode = st.selectbox("Source type", ["Upload file(s)", "URL"], label_visibility="collapsed")

    if load_mode == "Upload file(s)":
        uploaded = st.file_uploader("Drop files here", type=["pdf","txt","docx"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Load & Index") and uploaded:
            tmp_dir = Path("./tmp_uploads"); tmp_dir.mkdir(exist_ok=True)
            all_docs, names = [], []
            for f in uploaded:
                tmp_path = tmp_dir / f.name; tmp_path.write_bytes(f.read())
                all_docs.extend(load_file(str(tmp_path))); names.append(f.name)
            run_indexing(all_docs, names); shutil.rmtree(tmp_dir); st.rerun()
    else:
        url_input = st.text_input("Enter URL", placeholder="https://...")
        if st.button("Fetch & Index") and url_input:
            try:
                docs  = load_url(url_input)
                short = url_input[:40]+"..." if len(url_input)>40 else url_input
                run_indexing(docs, [short]); st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")
    if st.button("🗑 Clear conversation"):
        st.session_state.history = []; st.session_state.eval_log = []; st.rerun()

    if st.session_state.history:
        pdf_bytes = export_chat_pdf(st.session_state.history)
        st.download_button("💾 Export chat as PDF", data=pdf_bytes,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
# Top bar with eval toggle
title_col, toggle_col = st.columns([8, 2])
with title_col:
    st.markdown('<div class="librarium-title">Librarium</div>', unsafe_allow_html=True)
    st.markdown('<div class="librarium-subtitle">Ask anything. Find everything.</div>', unsafe_allow_html=True)
with toggle_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("📊 Eval Dashboard"):
        st.session_state.show_eval = not st.session_state.show_eval

st.markdown('<hr class="title-divider">', unsafe_allow_html=True)

# Split into chat + eval panel when open
if st.session_state.show_eval and st.session_state.eval_log:
    chat_col, eval_col = st.columns([6, 4])
else:
    chat_col = st.container()
    eval_col = None

with chat_col:
    if st.session_state.summary:
        st.markdown(f'<div class="summary-box"><div class="summary-label">Document Summary</div>{st.session_state.summary}</div>', unsafe_allow_html=True)

    if st.session_state.retriever is None:
        st.markdown('<div style="text-align:center;padding:4rem 2rem;color:#8b6f47;font-style:italic;font-size:1.1rem;">Upload a document or paste a URL in the sidebar to begin.</div>', unsafe_allow_html=True)
    else:
        for turn in st.session_state.history:
            st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{turn["question"]}</div>', unsafe_allow_html=True)
            cit_html = ""
            if turn.get("citations"):
                items    = "".join(f'<div class="citation-item">{c}</div>' for c in turn["citations"])
                cit_html = f'<div class="citations">{items}</div>'
            st.markdown(f'<div class="chat-label">Librarium</div><div class="chat-assistant">{turn["answer"]}{cit_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([5,1])
            with c1: question = st.text_input("", placeholder="Ask a question about your documents...", label_visibility="collapsed")
            with c2: submitted = st.form_submit_button("Ask →")

        if submitted and question.strip():
            answer_chain, rewrite_chain, _ = get_chains()

            with st.spinner("Consulting the library..."):
                t0                 = time.time()
                docs               = get_docs(question, st.session_state.history, rewrite_chain)
                context, citations = format_docs_with_citations(docs)

            # Stream the answer
            st.markdown('<div class="chat-label">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-user">{question}</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-label">Librarium</div>', unsafe_allow_html=True)

            stream_prompt = answer_chain.first  # the PromptTemplate
            llm, _    = get_llms()

            formatted = stream_prompt.invoke({
                "context":  context,
                "question": question,
                "history":  format_history(st.session_state.history)
            })

            answer = st.write_stream(llm.stream(formatted.text))
            elapsed   = round(time.time() - t0, 1)
            grounding = compute_grounding_score(answer, docs)

            # Show citations after streaming
            if citations:
                items    = "".join(f'<div class="citation-item">{c}</div>' for c in citations)
                st.markdown(f'<div class="citations">{items}</div>', unsafe_allow_html=True)

            st.session_state.history.append({"question": question, "answer": answer, "citations": citations})
            st.session_state.eval_log.append({
                "question":         question,
                "response_time":    elapsed,
                "grounding":        grounding,
                "sources_used":     len(set(c.split(" - ")[0] for c in citations)),
                "chunks_retrieved": len(docs),
            })
            st.rerun()

# ── EVAL PANEL ────────────────────────────────────────────────────────────────
if eval_col is not None:
    with eval_col:
        log = st.session_state.eval_log

        # Session stats
        avg_time      = round(sum(e["response_time"] for e in log) / len(log), 1)
        avg_grounding = round(sum(e["grounding"] for e in log) / len(log), 2)
        total_q       = len(log)
        avg_chunks    = round(sum(e["chunks_retrieved"] for e in log) / len(log), 1)

        st.markdown(f"""
        <div class="eval-panel">
            <div class="eval-title">📊 Session Overview</div>
            <div class="stat-grid">
                <div class="stat-box"><div class="stat-num">{total_q}</div><div class="stat-lbl">Queries</div></div>
                <div class="stat-box"><div class="stat-num">{avg_time}s</div><div class="stat-lbl">Avg response</div></div>
                <div class="stat-box"><div class="stat-num">{int(avg_grounding*100)}%</div><div class="stat-lbl">Avg grounding</div></div>
                <div class="stat-box"><div class="stat-num">{avg_chunks}</div><div class="stat-lbl">Avg chunks</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-query breakdown
        st.markdown('<div class="eval-panel"><div class="eval-title">Per Query Breakdown</div>', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(log), 1):
            bar_w  = int(entry["grounding"] * 80)
            q_short = entry["question"][:40] + "..." if len(entry["question"]) > 40 else entry["question"]
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;padding-bottom:0.8rem;border-bottom:1px dashed var(--border)">
                <div style="font-size:0.8rem;color:#c4a882;font-style:italic;margin-bottom:0.3rem">{q_short}</div>
                <div class="eval-row"><span>Response time</span><span class="eval-val">{entry["response_time"]}s</span></div>
                <div class="eval-row">
                    <span>Grounding</span>
                    <span class="eval-val">
                        {int(entry["grounding"]*100)}%
                        <span class="eval-bar-wrap"><span class="eval-bar-fill" style="width:{bar_w}px"></span></span>
                    </span>
                </div>
                <div class="eval-row"><span>Chunks retrieved</span><span class="eval-val">{entry["chunks_retrieved"]}</span></div>
                <div class="eval-row"><span>Sources used</span><span class="eval-val">{entry["sources_used"]}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)