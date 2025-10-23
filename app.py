import os
import shutil
import tempfile
from typing import List

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Optional LLM via Groq
try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

# Optional local CPU embeddings (fast & tiny, no API key needed)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# ---------- Constants / Folders ----------
UPLOAD_DIR = "uploads"
INDEX_DIR = "storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant for question answering over provided PDF documents. "
    "Use only the retrieved context to answer. If the answer isn't in the documents, say you don't know. "
    "When helpful, cite sources as [source: filename.pdf p.#] using the provided metadata. "
    "Keep answers concise and accurate."
)

# ---------- Small helpers ----------
def save_uploaded_pdfs(uploaded_files) -> List[str]:
    saved = []
    for uf in uploaded_files or []:
        if not uf.name.lower().endswith(".pdf"):
            continue
        dst_path = os.path.join(UPLOAD_DIR, os.path.basename(uf.name))
        with open(dst_path, "wb") as f:
            f.write(uf.read())
        saved.append(dst_path)
    return saved

def split_pdfs(paths: List[str]):
    docs = []
    for p in paths:
        loader = PyPDFLoader(p)
        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = os.path.basename(p)
        docs.extend(file_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(docs)

def build_index(paths: List[str]):
    docs = split_pdfs(paths)
    embeddings = FastEmbedEmbeddings()  # default: BAAI/bge-small-en-v1.5
    vs = FAISS.from_documents(docs, embeddings)
    # persist to disk for this Space runtime
    if os.path.exists(os.path.join(INDEX_DIR, "index")):
        shutil.rmtree(os.path.join(INDEX_DIR, "index"))
    vs.save_local(os.path.join(INDEX_DIR, "index"))
    return vs, len(docs), embeddings

def load_index():
    try:
        embeddings = FastEmbedEmbeddings()
        vs = FAISS.load_local(
            os.path.join(INDEX_DIR, "index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception:
        return None

def format_citations(chunks) -> str:
    cites, seen = [], set()
    for d in chunks:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append(f"[source: {src} p.{page+1}]" if page is not None else f"[source: {src}]")
    return " ".join(cites)

def get_llm():
    if os.getenv("GROQ_API_KEY") and HAS_GROQ:
        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        return ChatGroq(model_name=model, temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None

# ---------- UI ----------
st.set_page_config(page_title="GenAI PDF Q&A (RAG · LangChain)", page_icon="📄", layout="centered")

st.title("📄 GenAI PDF Q&A — RAG on LangChain (Streamlit)")
st.caption("Embeddings: FastEmbed • Vector DB: FAISS • LLM: Groq (preferred) or OpenAI")

with st.sidebar:
    st.header("1) Upload PDFs")
    uploaded = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.info(f"{len(uploaded)} file(s) selected.")
    build = st.button("🔧 Build Index", use_container_width=True)
    st.markdown("---")
    st.subheader("Secrets")
    st.write("Add keys in **Spaces → Settings → Variables**")
    st.code("GROQ_API_KEY=...  (preferred)\nOPENAI_API_KEY=...  (optional)\nGROQ_MODEL=llama-3.1-70b-versatile", language="bash")

# Keep vector store in session memory
if "vs" not in st.session_state:
    st.session_state.vs = load_index()

if build:
    saved_paths = save_uploaded_pdfs(uploaded)
    if not saved_paths:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Building vector index (FAISS + FastEmbed)..."):
            vs, n_chunks, _ = build_index(saved_paths)
            st.session_state.vs = vs
        st.success(f"Index ready: {len(saved_paths)} PDFs • {n_chunks} chunks")

# Chat section
st.header("2) Ask a question")
q = st.text_input("Your question about the PDFs:", placeholder="e.g., Summarize chapter 2's main points")

if st.button("💬 Ask", type="primary"):
    if not q.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vs is None:
        st.error("Please upload PDFs and click **Build Index** first.")
    else:
        llm = get_llm()
        if llm is None:
            st.error("No LLM provider configured. Set a GROQ_API_KEY (preferred) or OPENAI_API_KEY in Space secrets.")
        else:
            retriever = st.session_state.vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            with st.spinner("Retrieving context & generating answer..."):
                chunks = retriever.get_relevant_documents(q)
                if not chunks:
                    st.info("No relevant context found in the PDFs.")
                else:
                    context = "\n\n---\n\n".join([d.page_content for d in chunks][:6])
                    prompt = ChatPromptTemplate.from_messages(
                        [("system", SYSTEM_PROMPT),
                         ("human", "Question: {question}\n\nContext:\n{context}")]
                    )
                    msgs = prompt.format_messages(question=q, context=context)
                    resp = llm.invoke(msgs)
                    content = getattr(resp, "content", str(resp)).strip()
                    st.write(content)
                    cites = format_citations(chunks)
                    if cites:
                        st.caption(cites)

st.markdown("---")
st.markdown("**Tip:** If the Space sleeps, the on-disk FAISS index is rebuilt when you click *Build Index* again.")