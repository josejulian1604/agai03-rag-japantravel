"""
retriever.py
------------
Hybrid retriever with two-stage search:

  Stage 1 — Q/A collection:
    Search for a question similar to the user's query.
    If similarity score > THRESHOLD → return the stored answer directly.
    mode = "qa_direct"

  Stage 2 — Documents collection (fallback):
    If no Q/A match is confident enough, search in raw document chunks.
    Return top chunks as context for the LLM to generate an answer.
    mode = "vector_search"
"""

import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
QA_THRESHOLD = 0.75   # minimum similarity to use a Q/A direct answer
N_RESULTS    = 3      # number of document chunks to retrieve in fallback


class HybridRetriever:
    """
    Two-stage retriever for the Japan Travel RAG chatbot.

    Usage:
        retriever = HybridRetriever()
        result = retriever.retrieve("What airports serve Tokyo?")

        result = {
            "mode":    "qa_direct" | "vector_search",
            "answer":  str | None,     # direct answer (qa_direct only)
            "context": str | None,     # doc chunks joined (vector_search only)
            "sources": list[dict],     # [{url, title, score}, ...]
            "score":   float,          # top similarity score
        }
    """

    def __init__(self):
        # Connect to the persisted ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

        self.qa_collection   = client.get_collection(
            name="qa_pairs",
            embedding_function=embed_fn,
        )
        self.docs_collection = client.get_collection(
            name="documents",
            embedding_function=embed_fn,
        )

    # ── Stage 1: Q/A search ───────────────────────────────────────────────────
    def _search_qa(self, query: str) -> dict | None:
        """
        Search the Q/A collection for a question similar to the query.
        Returns a result dict if score > QA_THRESHOLD, else None.
        """
        results = self.qa_collection.query(
            query_texts=[query],
            n_results=1,
        )

        if not results["documents"][0]:
            return None

        # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite)
        # Convert to SIMILARITY: 1 - distance
        distance   = results["distances"][0][0]
        similarity = 1 - distance

        if similarity < QA_THRESHOLD:
            return None   # not confident enough — fall through to stage 2

        meta = results["metadatas"][0][0]

        return {
            "mode":    "qa_direct",
            "answer":  meta["answer"],
            "context": None,
            "sources": [{
                "url":   meta.get("source_url", ""),
                "title": meta.get("source_title", ""),
                "score": round(similarity, 3),
            }],
            "score": round(similarity, 3),
            "matched_question": results["documents"][0][0],
        }

    # ── Stage 2: Document search ──────────────────────────────────────────────
    def _search_docs(self, query: str) -> dict:
        """
        Search raw document chunks and return them as context.
        The LLM will use this context to generate an answer.
        """
        results = self.docs_collection.query(
            query_texts=[query],
            n_results=N_RESULTS,
        )

        chunks    = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        # Join chunks into a single context block for the LLM
        context_parts = []
        sources       = []
        seen_urls     = set()

        for chunk, meta, dist in zip(chunks, metas, distances):
            similarity = 1 - dist
            context_parts.append(chunk)

            url = meta.get("url", "")
            if url not in seen_urls:
                sources.append({
                    "url":   url,
                    "title": meta.get("title", ""),
                    "score": round(similarity, 3),
                })
                seen_urls.add(url)

        return {
            "mode":    "vector_search",
            "answer":  None,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "score":   round(1 - distances[0], 3),
        }

    # ── Public interface ──────────────────────────────────────────────────────
    def retrieve(self, query: str) -> dict:
        """
        Main retrieval method. Tries Q/A first, falls back to documents.
        Always returns a dict with: mode, answer, context, sources, score.
        """
        # Stage 1: try Q/A direct match
        qa_result = self._search_qa(query)
        if qa_result:
            return qa_result

        # Stage 2: fallback to document chunks
        return self._search_docs(query)


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "What airports serve Tokyo?",                    # should hit qa_direct
        "What is the best time to see cherry blossoms?", # should hit qa_direct
        "What's the atmosphere like walking in Gion?",   # likely vector_search
        "Tell me about traditional Japanese architecture",# likely vector_search
    ]

    print("🔍 Hybrid Retriever — Test\n" + "─"*55)

    for query in test_queries:
        result = retriever.retrieve(query)
        print(f"\nQ: {query}")
        print(f"   Mode  : {result['mode']}")
        print(f"   Score : {result['score']}")

        if result["mode"] == "qa_direct":
            print(f"   Answer: {result['answer'][:120]}...")
            print(f"   Match : {result.get('matched_question', '')[:80]}")
        else:
            print(f"   Context: {result['context'][:120]}...")

        if result["sources"]:
            print(f"   Source: {result['sources'][0]['title'][:50]}")

    print("\n" + "─"*55)