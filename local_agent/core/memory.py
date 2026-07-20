"""
Long-Term Memory backed by ChromaDB vector store.

Design:
  - Singleton pattern so all callers share the same connection.
  - Instance-level state (_client, _collection) to avoid class-level mutation.
  - Thread-safe lazy initialisation via threading.Lock.
  - Graceful fallback to an in-process dict when ChromaDB is not installed.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from pathlib import Path

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Singleton long-term memory backed by ChromaDB (or in-memory fallback)."""

    _instance: Optional["LongTermMemory"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LongTermMemory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    obj = super().__new__(cls)
                    obj._init_lock = threading.Lock()
                    obj._client = None
                    obj._collection = None
                    cls._instance = obj
        return cls._instance

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _get_collection(self):
        """Lazily initialise and return the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        with self._init_lock:
            if self._collection is not None:  # another thread may have initialised
                return self._collection
            try:
                import chromadb
                from local_agent.core.config import get_settings

                path = get_settings().memory_chroma_path
                Path(path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=path)
                self._collection = self._client.get_or_create_collection(
                    name="local_agent_memory",
                    metadata={"description": "LocalAgent long-term memory"},
                )
                logger.info("ChromaDB memory initialised at %s", path)
            except ImportError:
                logger.warning(
                    "chromadb not installed – using in-process fallback memory. "
                    "Install with: pip install chromadb"
                )
                self._collection = _InMemoryCollection()

        return self._collection

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def save(
        self,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Persist *content* and return a short unique ID."""
        col = self._get_collection()
        mem_id = str(uuid.uuid4())[:8]
        col.add(
            documents=[content],
            metadatas=[{
                "category": category,
                "tags": ",".join(tags or []),
                "content": content,
            }],
            ids=[mem_id],
        )
        return mem_id

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search; returns list of memory dicts sorted by relevance."""
        col = self._get_collection()
        try:
            results = col.query(query_texts=[query], n_results=n_results)
        except Exception as exc:
            logger.error("Memory search error: %s", exc)
            return []

        memories: List[Dict[str, Any]] = []
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for mem_id, doc, meta, dist in zip(ids, docs, metas, dists):
            memories.append({
                "id": mem_id,
                "content": meta.get("content", doc),
                "category": meta.get("category", "general"),
                "tags": [t for t in meta.get("tags", "").split(",") if t],
                "score": round(1.0 - float(dist), 4),
            })
        return memories

    def list_all(
        self,
        category: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List stored memories, optionally filtered by *category*."""
        col = self._get_collection()
        try:
            where = {"category": category} if category else None
            results = col.get(where=where, limit=limit)
        except Exception as exc:
            logger.error("Memory list error: %s", exc)
            return []

        memories: List[Dict[str, Any]] = []
        for mem_id, meta in zip(results.get("ids", []), results.get("metadatas", [])):
            memories.append({
                "id": mem_id,
                "content": meta.get("content", ""),
                "category": meta.get("category", "general"),
                "tags": [t for t in meta.get("tags", "").split(",") if t],
            })
        return memories

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID; returns True on success."""
        try:
            self._get_collection().delete(ids=[memory_id])
            return True
        except Exception as exc:
            logger.error("Memory delete error: %s", exc)
            return False

    def get_context(self, query: str, n_results: int = 3) -> str:
        """Return a formatted string of top-N relevant memories for prompt injection."""
        memories = self.search(query, n_results=n_results)
        if not memories:
            return ""
        return "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        with cls._lock:
            cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# In-process fallback (no ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryCollection:
    """Simple keyword-matching in-memory store, used when chromadb is absent."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def add(self, documents: List[str], metadatas: List[dict], ids: List[str]) -> None:
        for doc, meta, mid in zip(documents, metadatas, ids):
            self._data[mid] = {"document": doc, "metadata": meta}

    def query(self, query_texts: List[str], n_results: int) -> Dict[str, Any]:
        query = query_texts[0].lower()
        matches = [
            (mid, d)
            for mid, d in self._data.items()
            if any(word in d["document"].lower() for word in query.split())
        ]
        matches = matches[:n_results]
        return {
            "ids": [[m[0] for m in matches]],
            "documents": [[m[1]["document"] for m in matches]],
            "metadatas": [[m[1]["metadata"] for m in matches]],
            "distances": [[0.5] * len(matches)],
        }

    def get(self, where: Optional[dict] = None, limit: int = 20) -> Dict[str, Any]:
        results: Dict[str, Any] = {"ids": [], "metadatas": []}
        for mid, d in list(self._data.items())[:limit]:
            if where and not all(d["metadata"].get(k) == v for k, v in where.items()):
                continue
            results["ids"].append(mid)
            results["metadatas"].append(d["metadata"])
        return results

    def delete(self, ids: List[str]) -> None:
        for mid in ids:
            self._data.pop(mid, None)
