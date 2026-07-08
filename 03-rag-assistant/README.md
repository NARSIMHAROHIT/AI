# Project 3: RAG Personal Assistant

The chatbot in project 2 only knows what the model learned in training. This project gives it *my* knowledge: it answers questions from my own documents. This is RAG — Retrieval-Augmented Generation — and it's the single most common pattern in real-world LLM applications.

I'm building it in three stages:

- **Stage 1 (done): the RAG core** — documents in, grounded answers out
- **Stage 2 (done): MCP server** — retrieval exposed as tools any AI client can call
- **Stage 3 (done): UI** — a Streamlit chat interface, making it a personal assistant

## How RAG works

You can't paste your whole life into a prompt — context windows are finite. RAG solves this in two phases:

**Ingestion (`ingest.py`, run once):** read every file in `docs/`, split into overlapping ~800-character chunks, convert each chunk into a 384-number vector using a Hugging Face embedding model (`all-MiniLM-L6-v2`, runs locally, free). Vectors capture *meaning* — chunks about similar topics get similar numbers. Store them in ChromaDB.

**Retrieval (`ask.py`, every question):** embed the question with the same model, find the 4 chunks whose vectors are closest to the question's vector, paste just those chunks into the prompt, and have the Groq LLM answer from them — with instructions to say "I don't know" rather than guess beyond the context.

The magic is that similarity search works on meaning, not keywords: asking "what did I score on the flower model?" finds the iris accuracy chunk even though the words don't match.

## Run

```bash
pip install -r ../requirements.txt   # adds chromadb + sentence-transformers

# put your own .md/.txt files in docs/ (two samples included), then:
python ingest.py
python ask.py "What accuracy did the iris classifier get?"
python ask.py --show-chunks "What is Rohit building next?"   # peek at retrieval
python ask.py "What is the capital of France?"               # should refuse — not in docs
```

First run downloads the embedding model (~80MB). The `chroma_db/` folder it creates is gitignored — anyone can rebuild it with `ingest.py`.

## Stage 2: the MCP server

`mcp_server.py` wraps the same retrieval pipeline in the Model Context Protocol — a standard that lets any AI application discover and call my tools. Three tools are exposed: `search_docs` (raw chunk retrieval), `ask_docs` (full RAG answer via Groq), and `list_sources`.

The interesting shift: in stage 1, *my code* decided when to retrieve. With MCP, the *client's LLM* reads the tool descriptions and decides for itself when to search my documents — the same tool-calling pattern from project 1, but across a process boundary via a standard protocol.

Test it with the MCP Inspector (a debugging UI):

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

Or plug it into Claude Desktop — edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "personal-docs": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/TO/03-rag-assistant/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop, then ask Claude "what projects has Rohit built?" — it will call `search_docs` and answer from my files.

## Stage 3: the UI

`app.py` is the whole assistant in one Streamlit page: chat with memory (project 2's trimming strategy), grounded answers with a "Sources used" expander under each reply (stage 1's retrieval), and a sidebar to switch model tiers mid-conversation (project 1's idea). A toggle switches between document-grounded mode and general chat.

```bash
pip install streamlit
streamlit run app.py
```

One Streamlit quirk worth knowing: the script reruns top-to-bottom on *every* interaction, so the embedding model and DB connection are wrapped in `@st.cache_resource` (load once) and the conversation lives in `st.session_state` (survives reruns).

## Bonus: the MCP client agent

`agent_mcp.py` closes the loop. Project 1's agent had its tools hard-coded; this one *discovers* them. It reads `mcp_config.json`, launches every server listed, asks each for its tools, and hands them all to the Groq LLM. Ask a question, and the LLM decides which server's tool to call.

```bash
python agent_mcp.py --list                            # see discovered tools
python agent_mcp.py "What projects has Rohit built?"  # watch it call search_docs
```

The point of the config file: growing the agent means adding an entry, not writing code. Any community MCP server (filesystem, GitHub, Google Calendar, Gmail...) plugs in the same way:

```json
"servers": {
  "personal-docs": {"command": "python", "args": ["mcp_server.py"]},
  "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]}
}
```

## What I learned

RAG quality is mostly retrieval quality. If the right chunk isn't in the top 4, the LLM can't answer no matter how smart it is — which is why chunk size, overlap, and embedding model choice matter more than the generation model.
