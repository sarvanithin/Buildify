"""
RAG (Retrieval-Augmented Generation) system using Ollama nomic-embed-text.
Retrieves relevant US architectural standards to improve floor plan generation.
"""
import json
import math
import asyncio
import httpx
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
KB_PATH = Path(__file__).parent / "arch_knowledge.json"
CACHE_PATH = Path(__file__).parent / "embed_cache.json"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


async def _embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        return resp.json()["embedding"]


class RAGSystem:
    def __init__(self):
        self._chunks: list[dict] = []
        self._embeddings: dict[str, list[float]] = {}
        self._ready = False

    async def initialize(self):
        """Load knowledge base and compute/cache embeddings."""
        kb = json.loads(KB_PATH.read_text())
        self._chunks = kb["chunks"]

        # Load cached embeddings
        if CACHE_PATH.exists():
            self._embeddings = json.loads(CACHE_PATH.read_text())

        # Embed any chunks not yet in cache
        missing = [c for c in self._chunks if c["id"] not in self._embeddings]
        if missing:
            print(f"[RAG] Computing embeddings for {len(missing)} new chunks...")
            for chunk in missing:
                try:
                    vec = await _embed(chunk["text"])
                    self._embeddings[chunk["id"]] = vec
                except Exception as e:
                    print(f"[RAG] Embed failed for {chunk['id']}: {e}")

            CACHE_PATH.write_text(json.dumps(self._embeddings))
            print(f"[RAG] Cached {len(self._embeddings)} embeddings.")

        self._ready = True
        print(f"[RAG] Ready. {len(self._chunks)} chunks loaded.")

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top-k most relevant knowledge chunks for the query."""
        if not self._ready or not self._embeddings:
            return []

        try:
            query_vec = await _embed(query)
        except Exception:
            return []

        scored = []
        for chunk in self._chunks:
            if chunk["id"] in self._embeddings:
                sim = _cosine_sim(query_vec, self._embeddings[chunk["id"]])
                scored.append((sim, chunk["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]


# Singleton instance
rag = RAGSystem()
