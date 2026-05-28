"""
vector_store.py
---------------
Builds the ChromaDB vector store from two sources:
  1. Scraped pages (data/raw/) → chunked and embedded → "documents" collection
  2. Q/A pairs (data/qa_dataset.csv) → embedded → "qa_pairs" collection

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, local, no API needed)
Vector DB:  ChromaDB (persisted to disk at chroma_db/)
"""

import os
import json
import csv
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
QA_CSV        = "data/qa_dataset.csv"
CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # free, runs locally, 384 dimensions

# Chunking settings
CHUNK_SIZE    = 512     # characters per chunk
CHUNK_OVERLAP = 80      # overlap between chunks to preserve context


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping chunks.

    Why overlap? So that sentences at chunk boundaries aren't lost.
    Example with overlap=3:
      Chunk 1: "A B C D E"
      Chunk 2: "D E F G H"   ← D and E repeated = context preserved
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to cut at a paragraph or sentence boundary
        if end < len(text):
            # Prefer cutting at double newline (paragraph)
            para_break = chunk.rfind("\n\n")
            if para_break > chunk_size // 2:
                end   = start + para_break
                chunk = text[start:end]
            else:
                # Fall back to sentence boundary (period + space)
                sent_break = chunk.rfind(". ")
                if sent_break > chunk_size // 2:
                    end   = start + sent_break + 1
                    chunk = text[start:end]

        chunk = chunk.strip()
        if len(chunk) > 50:          # skip very small chunks
            chunks.append(chunk)

        start = end - overlap        # move forward with overlap

    return chunks


# ── Load data ─────────────────────────────────────────────────────────────────
def load_pages() -> list:
    """Load all scraped pages from data/raw/."""
    pages = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
                pages.append(json.load(f))
    return pages


def load_qa_pairs() -> list:
    """Load Q/A pairs from CSV."""
    pairs = []
    with open(QA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append(row)
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────
def build_vector_store():

    # 1. Initialize ChromaDB client (persistent — survives between runs)
    print("🗄️  Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 2. Embedding function — runs locally with sentence-transformers
    #    First run downloads the model (~90MB), subsequent runs use cache
    print(f"🤖 Loading embedding model: {EMBED_MODEL}")
    print("   (First run downloads ~90MB — subsequent runs are instant)\n")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # 3. Create (or reset) collections
    #    delete_collection won't error if it doesn't exist
    for name in ["documents", "qa_pairs"]:
        try:
            client.delete_collection(name)
        except:
            pass

    docs_collection = client.create_collection(
        name="documents",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for text
    )

    qa_collection = client.create_collection(
        name="qa_pairs",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    print("✅ Collections created: 'documents' + 'qa_pairs'\n")

    # ── 4. Index scraped pages into "documents" collection ────────────────────
    pages = load_pages()
    print(f"📄 Indexing {len(pages)} pages into 'documents'...")

    total_chunks = 0
    chunk_ids    = []
    chunk_texts  = []
    chunk_metas  = []

    for page in tqdm(pages, desc="Chunking pages"):
        content = page.get("content", "")
        url     = page.get("url", "")
        title   = page.get("title", "")

        if not content:
            continue

        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_ids.append(f"{url}__chunk_{i}")
            chunk_texts.append(chunk)
            chunk_metas.append({
                "url":   url,
                "title": title,
                "chunk": i,
            })
            total_chunks += 1

    # Add to ChromaDB in batches (avoids memory issues with large datasets)
    BATCH = 50
    for i in tqdm(range(0, len(chunk_texts), BATCH), desc="Embedding docs"):
        docs_collection.add(
            ids       = chunk_ids[i:i+BATCH],
            documents = chunk_texts[i:i+BATCH],
            metadatas = chunk_metas[i:i+BATCH],
        )

    print(f"✅ Documents indexed: {total_chunks} chunks from {len(pages)} pages\n")

    # ── 5. Index Q/A pairs into "qa_pairs" collection ─────────────────────────
    qa_pairs = load_qa_pairs()
    print(f"❓ Indexing {len(qa_pairs)} Q/A pairs into 'qa_pairs'...")

    qa_ids   = []
    qa_texts = []   # we embed the QUESTION (what users will search with)
    qa_metas = []   # we store the ANSWER in metadata

    for i, pair in enumerate(qa_pairs):
        qa_ids.append(f"qa_{i:04d}")
        qa_texts.append(pair["question"])    # embed the question
        qa_metas.append({
            "answer":       pair["answer"],
            "source_url":   pair.get("source_url", ""),
            "source_title": pair.get("source_title", ""),
        })

    for i in tqdm(range(0, len(qa_texts), BATCH), desc="Embedding Q/A"):
        qa_collection.add(
            ids       = qa_ids[i:i+BATCH],
            documents = qa_texts[i:i+BATCH],
            metadatas = qa_metas[i:i+BATCH],
        )

    print(f"✅ Q/A pairs indexed: {len(qa_pairs)} pairs\n")

    # ── 6. Verification ───────────────────────────────────────────────────────
    print("🔍 Verification — test query: 'What airports serve Tokyo?'")
    results = qa_collection.query(
        query_texts=["What airports serve Tokyo?"],
        n_results=2,
    )

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"\n  Q: {doc}")
        print(f"  A: {meta['answer'][:100]}")
        print(f"  Score: {1 - dist:.3f}")   # convert distance to similarity

    print(f"\n{'─'*55}")
    print(f"✅  Vector store built successfully!")
    print(f"    Documents : {total_chunks} chunks")
    print(f"    Q/A pairs : {len(qa_pairs)}")
    print(f"    Location  : {CHROMA_DIR}/")
    print(f"{'─'*55}")


if __name__ == "__main__":
    build_vector_store()