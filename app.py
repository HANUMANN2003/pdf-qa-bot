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

st.set_page_config(page_title="PDF Q&A System", page_icon="📄", layout="centered")
st.title("📄 PDF Question Answering System")
st.caption("Upload a PDF, ask questions, get answers with source references.")

# Load Groq API key: try Streamlit secrets first (cloud), then env var (local .env)
api_key = None
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Upload a PDF\n"
        "2. We split it into chunks\n"
        "3. Each chunk is embedded into vectors (free, local model)\n"
        "4. Stored in a local ChromaDB\n"
        "5. Your question retrieves the closest chunks\n"
        "6. Groq's LLM answers using only those chunks"
    )

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.info("Get a free key at https://console.groq.com/keys")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Loads a free, local embedding model (downloads once, then cached)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes, file_name):
    """Loads, chunks, embeds, and stores the PDF in ChromaDB. Cached so it
    only runs once per unique uploaded file."""
    # Save uploaded bytes to a temp file because PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{abs(hash(file_name))}",
    )

    os.unlink(tmp_path)
    return vectorstore, len(documents), len(chunks)


if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Reading and indexing your PDF... (first run downloads the embedding model, ~1 min)"):
        vectorstore, num_pages, num_chunks = build_vectorstore(file_bytes, uploaded_file.name)

    st.success(f"Indexed {num_pages} pages → {num_chunks} chunks. Ready for questions!")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})

        st.markdown("### Answer")
        st.write(result["result"])

        st.markdown("### Sources")
        for i, doc in enumerate(result["source_documents"], start=1):
            page = doc.metadata.get("page", "unknown")
            with st.expander(f"Source {i} — Page {page}"):
                st.write(doc.page_content)
else:
    st.info("Upload a PDF above to get started.")
