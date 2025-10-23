import os
import shutil
from typing import List

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Optional OpenAI via LangChain
from langchain_openai import ChatOpenAI

# Direct Groq SDK (no langchain_groq)
try:
    from groq import Groq
    HAS_GROQ_SDK = True
except Exception:
    HAS_GROQ_SDK = False


# ---------------------- Constants / Folders ----------------------
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


# ---------------------- Helpers ----------------------
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
    embeddings = FastEmbedEmbeddings()  # BAAI/bge-small-en-v1.5 (CPU, no key)
    vs = FAISS.from_documents(docs, embeddings)
    target = os.path.join(INDEX_DIR, "index")
    if os.path.exists(target):
        shutil.rmtree(target)
    vs.save_local(target)
    return vs, len(docs)


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


# ---------------------- LLM dispatch ----------------------
def generate_with_groq(system_prompt: str, question: str, context: str) -> str:
    """
    Call Groq directly using the groq SDK. Requires GROQ_API_KEY.
    Includes model fallbacks so decommissioned models won't crash the app.
    """
    if not (HAS_GROQ_SDK and os.getenv("GROQ_API_KEY")):
        return ""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Preferred model from env plus robust fallbacks (all currently supported)
    candidates = [
        (os.getenv("GROQ_MODEL") or "").strip() or None,
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    # dedupe & drop Nones
    seen, models = set(), []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            models.append(m)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]

    last_err = None
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # try next candidate
            last_err = e
            continue

    return f"LLM error: {last_err}"


def generate_with_openai(system_prompt: str, question: str, context: str) -> str:
    """Fallback via LangChain OpenAI (requires OPENAI_API_KEY)."""
    if not os.getenv("OPENAI_API_KEY"):
        return ""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
         ("human", "Question: {question}\n\nContext:\n{context}")]
    )
    msgs = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp)).strip()


# ---------------------- Streamlit UI ----------------------
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
    st.subheader("Secrets / Models")
    st.write("Add keys in **Spaces → Settings → Variables & secrets**")
    st.code(
        "GROQ_API_KEY=...\n"
        "GROQ_MODEL=llama-3.1-8b-instant  # recommended\n"
        "# Optional fallback:\n"
        "OPENAI_API_KEY=...\n",
        language="bash",
    )

if "vs" not in st.session_state:
    st.session_state.vs = load_index()

if build:
    saved_paths = save_uploaded_pdfs(uploaded)
    if not saved_paths:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Building vector index (FAISS + FastEmbed)..."):
            vs, n_chunks = build_index(saved_paths)
            st.session_state.vs = vs
        st.success(f"Index ready: {len(saved_paths)} PDFs • {n_chunks} chunks")

st.header("2) Ask a question")
q = st.text_input("Your question about the PDFs:", placeholder="e.g., Summarize chapter 2's main points")

if st.button("💬 Ask", type="primary"):
    if not q.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vs is None:
        st.error("Please upload PDFs and click **Build Index** first.")
    else:
        retriever = st.session_state.vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        with st.spinner("Retrieving context & generating answer..."):
            chunks = retriever.get_relevant_documents(q)
            if not chunks:
                st.info("No relevant context found in the PDFs.")
            else:
                context = "\n\n---\n\n".join([d.page_content for d in chunks][:6])

                # Prefer Groq; fallback to OpenAI
                answer = generate_with_groq(SYSTEM_PROMPT, q, context)
                if not answer:
                    answer = generate_with_openai(SYSTEM_PROMPT, q, context)
                if not answer:
                    st.error("No LLM provider configured. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY in Space secrets.")
                else:
                    st.write(answer)
                    cites = format_citations(chunks)
                    if cites:
                        st.caption(cites)

st.markdown("---")
st.markdown("**Tip:** If the Space sleeps, click **Build Index** again after uploading PDFs to rebuild FAISS.")
