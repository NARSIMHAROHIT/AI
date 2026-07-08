"""RAG stage 1b: answer questions using your own documents.

Pipeline: embed the question -> find the most similar chunks in ChromaDB
-> paste those chunks into the prompt -> let the Groq LLM answer from them.

Usage:
    python ask.py "What projects has Rohit built?"
    python ask.py --tier smart --show-chunks "..."
"""

import argparse
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6  # how many chunks to retrieve

MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "openai/gpt-oss-20b",
    "smart": "llama-3.3-70b-versatile",
}

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the context doesn't contain the answer, say you don't know — do not guess. "
    "Mention which source file(s) you used."
)


def retrieve(question: str) -> list[dict]:
    """Find the TOP_K chunks most similar in meaning to the question."""
    embedder = SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode([question]).tolist()

    collection = chromadb.PersistentClient(path=DB_DIR).get_collection("docs")
    hits = collection.query(query_embeddings=q_vec, n_results=TOP_K)

    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(hits["documents"][0], hits["metadatas"][0])
    ]


def answer(question: str, tier: str = "fast", show_chunks: bool = False) -> str:
    chunks = retrieve(question)

    if show_chunks:
        for i, c in enumerate(chunks, 1):
            print(f"--- chunk {i} (from {c['source']}) ---\n{c['text'][:200]}...\n")

    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    llm = ChatGroq(model=MODELS[tier], temperature=0)
    response = llm.invoke([
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    return response.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions about your documents")
    parser.add_argument("question")
    parser.add_argument("--tier", choices=MODELS, default="fast")
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the retrieved chunks before the answer")
    args = parser.parse_args()

    print(answer(args.question, args.tier, args.show_chunks))
