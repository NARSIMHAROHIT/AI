"""RAG stage 3: the personal assistant UI.

Everything from the previous projects, behind a chat interface:
  - conversation memory (project 2)
  - RAG over my documents (stage 1)
  - model tier switching (project 1)
Plus: add data sources (files/URLs) from the UI, edit the system prompt,
and tune temperature.

Run:
    streamlit run app.py
"""

import os
import re
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Streamlit Cloud secrets fallback (locally, .env covers it)
try:
    if not os.getenv("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"
DB_DIR = str(HERE / "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "openai/gpt-oss-20b",
    "smart": "llama-3.3-70b-versatile",
    "max": "openai/gpt-oss-120b",
}
MAX_TURNS = 10

DEFAULT_SYSTEM_PROMPT = (
    """You are a helpful, polite, and professional AI assistant.

Your primary responsibility is to answer only questions that are relevant to your assigned domain or the provided context.

Guidelines:

* Always respond in a polite, respectful, and friendly tone.
* Answer questions accurately and concisely when they are within the provided context.
* If a question is outside your scope, context, or knowledge base, do not attempt to guess or fabricate an answer.
* Instead, politely decline by saying that the question is outside your scope. For example:
    * “I’m sorry, but that question is outside the scope of what I can help with.”
    * “I can only assist with questions related to the provided context.”
    * “I don’t have enough context to answer that question. Please ask something related to the available information.”
* Never generate misleading or hallucinated information.
* If the user’s request is ambiguous, ask a clarifying question before answering.
* Stay focused on the current context and avoid discussing unrelated topics.
* Be consistent, courteous, and professional in every response."""
)


@st.cache_resource  # load heavy things once, not on every rerun
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection("docs")
    except Exception:
        # No index yet (fresh clone / cloud deploy): build it from docs/
        import ingest
        ingest.main()
        return client.get_collection("docs")


def retrieve(question: str, k: int = 4) -> list[dict]:
    q_vec = get_embedder().encode([question]).tolist()
    hits = get_collection().query(query_embeddings=q_vec, n_results=k)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(hits["documents"][0], hits["metadatas"][0])
    ]


def fetch_url_as_text(url: str) -> str:
    """Download a webpage and strip it to readable text."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()


def rebuild_index():
    """Re-ingest docs/ and refresh the cached collection."""
    import ingest
    ingest.main()
    get_collection.clear()  # next retrieve() reopens the fresh collection


# --- Sidebar: settings ---
st.sidebar.title("Personal Assistant")
tier = st.sidebar.radio("Model tier", list(MODELS), index=0,
                        format_func=lambda t: f"{t} ({MODELS[t]})")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                                help="Low = focused and factual, high = creative")
use_rag = st.sidebar.toggle("Answer from my documents", value=True)

with st.sidebar.expander("System prompt"):
    system_prompt = st.text_area("Instructions for the assistant",
                                 value=DEFAULT_SYSTEM_PROMPT, height=180)

with st.sidebar.expander("Add data sources"):
    uploads = st.file_uploader("Upload files", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    url = st.text_input("Or add a webpage URL", placeholder="https://...")
    if st.button("Add & rebuild index"):
        added = []
        for up in uploads or []:
            (DOCS_DIR / up.name).write_bytes(up.getbuffer())
            added.append(up.name)
        if url.strip():
            try:
                text = fetch_url_as_text(url.strip())
                name = re.sub(r"\W+", "_", url.split("//")[-1])[:60] + ".txt"
                (DOCS_DIR / name).write_text(text, encoding="utf-8")
                added.append(name)
            except Exception as exc:
                st.error(f"Couldn't fetch URL: {exc}")
        if added:
            with st.spinner("Chunking and embedding..."):
                rebuild_index()
            st.success(f"Added {', '.join(added)} — index rebuilt.")
        else:
            st.info("Upload a file or enter a URL first.")

if st.sidebar.button("Clear conversation"):
    st.session_state.history = []

st.sidebar.caption("Turn the toggle off for general chat; on to ground "
                   "answers in your documents.")

# --- Chat state (Streamlit reruns the whole script on every interaction) ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Machine learning assistant")
st.caption("Ask questions about your documents, or just chat.")

# Replay past messages so the conversation stays on screen
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Handle a new message ---
if question := st.chat_input("Type your question..."):
    st.session_state.history.append(HumanMessage(question))
    with st.chat_message("user"):
        st.markdown(question)

    if use_rag:
        chunks = retrieve(question)
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        system = f"{system_prompt}\n\nContext:\n{context}"
    else:
        system = system_prompt

    llm = ChatGroq(model=MODELS[tier], temperature=temperature)
    messages = [SystemMessage(system)] + st.session_state.history[-(MAX_TURNS * 2):]

    with st.chat_message("assistant"):
        response = llm.invoke(messages)
        st.markdown(response.content)
        if use_rag:
            with st.expander("Sources used"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**\n\n> {c['text'][:300]}...")

    st.session_state.history.append(AIMessage(response.content))
