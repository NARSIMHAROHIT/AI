"""RAG stage 3: the personal assistant UI.

Everything from the previous projects, behind a chat interface:
  - conversation memory (project 2)
  - RAG over my documents (stage 1)
  - model tier switching (project 1)

Run:
    streamlit run app.py
"""

from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "openai/gpt-oss-20b",
    "smart": "llama-3.3-70b-versatile",
}
MAX_TURNS = 10


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


# --- Sidebar: settings ---
st.sidebar.title("Personal Assistant")
tier = st.sidebar.radio("Model tier", list(MODELS), index=0,
                        format_func=lambda t: f"{t} ({MODELS[t]})")
use_rag = st.sidebar.toggle("Answer from my documents", value=True)
if st.sidebar.button("Clear conversation"):
    st.session_state.history = []

st.sidebar.caption("Turn the toggle off for general chat; on to ground "
                   "answers in the docs/ folder.")

# --- Chat state (Streamlit reruns the whole script on every interaction) ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Ask me anything")

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
        system = ("Answer using ONLY the provided context and conversation "
                  "history. If the answer isn't there, say you don't know. "
                  f"Cite source files.\n\nContext:\n{context}")
    else:
        system = "You are a helpful, concise assistant."

    llm = ChatGroq(model=MODELS[tier], temperature=0.3)
    messages = [SystemMessage(system)] + st.session_state.history[-(MAX_TURNS * 2):]

    with st.chat_message("assistant"):
        response = llm.invoke(messages)
        st.markdown(response.content)
        if use_rag:
            with st.expander("Sources used"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**\n\n> {c['text'][:300]}...")

    st.session_state.history.append(AIMessage(response.content))
