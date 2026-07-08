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

import asyncio
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
    """You are a helpful, polite, and professional AI assistant that answers STRICTLY from the provided context documents.

Strict rules — no exceptions:

* Answer ONLY using information present in the provided context. Never use your own general knowledge, even if you are confident.
* Every answer must cite the source file(s) it came from.
* If the context does not contain the answer, reply exactly: "I don't have that in the knowledge base. Try adding a relevant document, or rephrase your question."
* Never guess, never fill gaps, never fabricate. Partial answers are fine if clearly marked as partial.
* Never attribute work described in reference material to Rohit unless his own notes say so.
* If the user's request is ambiguous, ask a clarifying question before answering.
* Be concise, consistent,friendly and explain the answer to the user clearly.
* If the question has multiple parts, answer every part the context covers, and use the refusal line only for the specific parts it doesn't."""
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
agent_mode = st.sidebar.toggle(
    "Agent mode (MCP tools)", value=False,
    help="The agent discovers tools from mcp_config.json (docs search, Gmail, "
         "Maps...) and decides which to call. Slower per question.")

with st.sidebar.expander("Your API keys (optional, session-only)"):
    st.caption("Keys live only in this browser session's memory — never "
               "stored, logged, or shared. Closing the tab erases them.")
    user_maps_key = st.text_input("Google Maps API key", type="password",
                                  help="Enables the Maps tools in Agent mode")
    user_groq_key = st.text_input("Your own Groq API key", type="password",
                                  help="Use your free key from console.groq.com "
                                       "instead of the site owner's")

# Session-only env for MCP servers; empty strings are dropped
session_env = {k: v for k, v in {
    "GOOGLE_MAPS_API_KEY": user_maps_key.strip(),
    "GROQ_API_KEY": user_groq_key.strip(),
}.items() if v}

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

st.sidebar.caption("Answers come strictly from your documents — the "
                   "assistant refuses rather than guesses.")

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

    if agent_mode:
        # --- Agent mode: let the LLM discover and call MCP tools ---
        from agent_mcp import run_agent

        trace = []
        with st.chat_message("assistant"):
            with st.spinner("Agent working — connecting to tools..."):
                past = st.session_state.history[:-1][-(MAX_TURNS * 2):]
                answer_text = asyncio.run(
                    run_agent(question, tier, history=past, trace=trace,
                              extra_env=session_env)
                )
            st.markdown(answer_text)
            if trace:
                with st.expander("Tool calls"):
                    for t in trace:
                        st.markdown(f"`{t['call']}`\n\n> {t['result'][:300]}...")
        st.session_state.history.append(AIMessage(answer_text))
    else:
        # --- Chat mode: documents are always retrieved; the prompt decides
        #     whether they're relevant enough to use ---
        chunks = retrieve(question)
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        system = f"{system_prompt}\n\nContext:\n{context}"

        llm = ChatGroq(model=MODELS[tier], temperature=temperature,
                       api_key=session_env.get("GROQ_API_KEY") or None)
        messages = [SystemMessage(system)] + st.session_state.history[-(MAX_TURNS * 2):]

        with st.chat_message("assistant"):
            response = llm.invoke(messages)
            st.markdown(response.content)
            with st.expander("Sources used"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**\n\n> {c['text'][:300]}...")

        st.session_state.history.append(AIMessage(response.content))
