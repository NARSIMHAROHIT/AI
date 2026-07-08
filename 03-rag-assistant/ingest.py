"""RAG stage 1a: ingest documents into a vector database.

Pipeline: read docs/ -> split into chunks -> embed each chunk with a
Hugging Face model (runs locally, free) -> store vectors in ChromaDB.

Run this once, and again whenever you add or change documents:
    python ingest.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent          # absolute paths: works no matter the cwd
DOCS_DIR = HERE / "docs"
DB_DIR = str(HERE / "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"  # Hugging Face model, ~80MB, downloads on first run
CHUNK_SIZE = 800    # characters per chunk
OVERLAP = 150       # chunks share edges so ideas aren't cut mid-sentence


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def main():
    files = sorted(DOCS_DIR.glob("*.md")) + sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        print(f"No .md or .txt files found in {DOCS_DIR}/ — add some and rerun.")
        return

    # Read + chunk
    all_chunks, metadatas = [], []
    for f in files:
        chunks = chunk_text(f.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)
        metadatas.extend({"source": f.name, "chunk": i} for i in range(len(chunks)))
        print(f"{f.name}: {len(chunks)} chunks")

    # Embed (each chunk becomes a 384-dim vector capturing its meaning)
    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Store (recreate the collection so re-ingesting stays in sync)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    collection = client.create_collection("docs")
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(all_chunks))],
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} chunks in {DB_DIR}/")


if __name__ == "__main__":
    main()
