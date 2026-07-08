# Project 3: RAG Personal Assistant

The chatbot in project 2 only knows what the model learned in training. This project gives it *my* knowledge: it answers questions from my own documents. This is RAG — Retrieval-Augmented Generation — and it's the single most common pattern in real-world LLM applications.

I'm building it in three stages:

- **Stage 1 (this code): the RAG core** — documents in, grounded answers out
- **Stage 2: MCP server** — expose retrieval as tools any AI client can call
- **Stage 3: UI** — a proper chat interface, making it a personal assistant

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

## What I learned

RAG quality is mostly retrieval quality. If the right chunk isn't in the top 4, the LLM can't answer no matter how smart it is — which is why chunk size, overlap, and embedding model choice matter more than the generation model.
