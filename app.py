import os
import sys
import tempfile

# --- Fix for Streamlit Cloud: it ships an old SQLite version that
# ChromaDB rejects. This swaps in a newer bundled SQLite before
# chromadb is imported anywhere. Safe to keep even if not needed locally.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="DocQuery — PDF Intelligence", page_icon="📑", layout="centered")

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0E1525;
    color: #E4E7EC;
}

/* Hero */
.dq-hero {
    padding: 2.2rem 0 1.4rem 0;
    border-bottom: 1px solid #232B3D;
    margin-bottom: 1.6rem;
}
.dq-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #C9962C;
    margin-bottom: 0.5rem;
}
.dq-title {
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #F4F1EA;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.dq-sub {
    font-size: 0.95rem;
    color: #9CA8B8;
    max-width: 540px;
    line-height: 1.5;
}

/* Stat strip */
.dq-stats {
    display: flex;
    gap: 1.6rem;
    margin-top: 1.2rem;
}
.dq-stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #4A8B8C;
}
.dq-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7686;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161D2E;
    border: 1px dashed #2C3650;
    border-radius: 10px;
    padding: 0.6rem;
}

/* Text input */
.stTextInput input {
    background: #161D2E;
    border: 1px solid #2C3650;
    border-radius: 8px;
    color: #F4F1EA;
    font-size: 0.98rem;
}
.stTextInput input:focus {
    border-color: #C9962C;
    box-shadow: 0 0 0 1px #C9962C;
}

/* Answer card */
.dq-answer-card {
    background: #F4F1EA;
    color: #1A1F2E;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin: 1rem 0 1.4rem 0;
    border-left: 4px solid #C9962C;
}
.dq-answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8A6A1F;
    margin-bottom: 0.5rem;
}
.dq-answer-text {
    font-size: 1rem;
    line-height: 1.6;
}

/* Source citation cards */
.dq-source-card {
    background: #161D2E;
    border: 1px solid #232B3D;
    border-left: 3px solid #4A8B8C;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
}
.dq-source-tab {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #0E1525;
    background: #4A8B8C;
    border-radius: 4px;
    padding: 0.18rem 0.5rem;
    white-space: nowrap;
    margin-top: 0.1rem;
}
.dq-source-text {
    font-size: 0.85rem;
    color: #9CA8B8;
    line-height: 1.5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #232B3D;
}
.dq-side-title {
    font-family: 'Lora', serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: #F4F1EA;
    margin-bottom: 0.6rem;
}
.dq-side-step {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #9CA8B8;
    line-height: 1.9;
}
.dq-side-num {
    color: #C9962C;
}

footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------
api_key = None
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.markdown('<div class="dq-side-title">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="dq-side-step"><span class="dq-side-num">01</span> Upload a PDF</div>
    <div class="dq-side-step"><span class="dq-side-num">02</span> Text is split into chunks</div>
    <div class="dq-side-step"><span class="dq-side-num">03</span> Chunks embedded into vectors</div>
    <div class="dq-side-step"><span class="dq-side-num">04</span> Stored in ChromaDB</div>
    <div class="dq-side-step"><span class="dq-side-num">05</span> Question retrieves top matches</div>
    <div class="dq-side-step"><span class="dq-side-num">06</span> LLM answers from evidence</div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("---")
        api_key = st.text_input("Groq API Key", type="password")

    st.markdown("---")
    st.markdown(
        '<div class="dq-side-step" style="color:#6B7686;">'
        'Stack: LangChain · ChromaDB · Groq (Llama 3.3 70B) · '
        'HuggingFace embeddings · Streamlit</div>',
        unsafe_allow_html=True,
    )

if not api_key:
    st.warning("Add a Groq API key to continue. Get a free one at console.groq.com/keys")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown("""
<div class="dq-hero">
    <div class="dq-eyebrow">Retrieval-Augmented Document Q&A</div>
    <div class="dq-title">📑 DocQuery</div>
    <div class="dq-sub">Upload a PDF and ask questions in plain English. Every
    answer is grounded in the document and traced back to its exact source page —
    no hallucinated claims, no guessing.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core logic
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    os.unlink(tmp_path)

    if not chunks:
        return None, len(documents), 0

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{abs(hash(file_name))}",
    )

    return vectorstore, len(documents), len(chunks)


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Indexing document — splitting, embedding, storing in ChromaDB..."):
        vectorstore, num_pages, num_chunks = build_vectorstore(file_bytes, uploaded_file.name)

    if vectorstore is None:
        st.error(
            "No readable text was found in this PDF. This usually means it's a "
            "**scanned document** (pages are images, not real text) rather than a "
            "text-based PDF. Try a different file, or run OCR on it first to "
            "convert the scanned pages into selectable text."
        )
        st.stop()

    st.markdown(f"""
    <div class="dq-stats">
        <div><div class="dq-stat-num">{num_pages}</div><div class="dq-stat-label">Pages</div></div>
        <div><div class="dq-stat-num">{num_chunks}</div><div class="dq-stat-label">Chunks</div></div>
        <div><div class="dq-stat-num">384d</div><div class="dq-stat-label">Vector Dim</div></div>
        <div><div class="dq-stat-num">k=3</div><div class="dq-stat-label">Retrieved</div></div>
    </div>
    <br>
    """, unsafe_allow_html=True)

    question = st.text_input("Ask a question", placeholder="e.g. What are the key findings in this document?", label_visibility="collapsed")

    if question:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        with st.spinner("Retrieving evidence and generating answer..."):
            result = qa_chain.invoke({"query": question})

        st.markdown(f"""
        <div class="dq-answer-card">
            <div class="dq-answer-label">Answer</div>
            <div class="dq-answer-text">{result["result"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dq-answer-label" style="color:#9CA8B8;">Source evidence</div><br>', unsafe_allow_html=True)

        for doc in result["source_documents"]:
            page = doc.metadata.get("page", "—")
            snippet = doc.page_content[:220].replace("\n", " ").strip() + "..."
            st.markdown(f"""
            <div class="dq-source-card">
                <div class="dq-source-tab">PAGE {page}</div>
                <div class="dq-source-text">{snippet}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="color:#6B7686; font-size:0.9rem; padding-top:0.5rem;">'
        'No document loaded. Upload a PDF above to begin.</div>',
        unsafe_allow_html=True,
    )
