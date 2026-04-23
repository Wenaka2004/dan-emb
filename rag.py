"""RAG retrieval service for Danbooru tag wiki knowledge base."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, EMBEDDING_MODEL, INDEX_DIR, DEFAULT_TOP_K

# ── Embedding API ───────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns normalized vector."""
    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text, "encoding_format": "float"},
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec


# ── RAG Service ─────────────────────────────────────────────────────────────

class DanbooruRAG:
    def __init__(self, index_dir: str | Path = INDEX_DIR):
        index_dir = Path(index_dir)
        t0 = time.perf_counter()

        # Load embeddings: prefer fp16 npz (smallest), then fp32 npz, then npy
        npz_f16 = index_dir / "embeddings_fp16.npz"
        npz_f32 = index_dir / "embeddings.npz"
        npy_path = index_dir / "embeddings.npy"
        if npz_f16.exists():
            self.embeddings = np.load(npz_f16)["embeddings"].astype(np.float32)
        elif npz_f32.exists():
            self.embeddings = np.load(npz_f32)["embeddings"].astype(np.float32)
        elif npy_path.exists():
            self.embeddings = np.load(npy_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"No embeddings file found in {index_dir}")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings /= norms

        self.meta = pd.read_parquet(index_dir / "metadata.parquet")

        cp_path = index_dir / "char_copyright.json"
        if cp_path.exists():
            with open(cp_path, encoding="utf-8") as f:
                self.char_copyright = json.load(f)
        else:
            self.char_copyright = {}

        self.tag_to_idx = {tag: i for i, tag in enumerate(self.meta["tag"])}

        elapsed = time.perf_counter() - t0
        print(f"RAG loaded: {len(self.embeddings)} vectors, {self.embeddings.shape[1]}d, {elapsed:.1f}s")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        expand_copyright: bool = True,
        category_filter: list[str] | None = None,
    ) -> list[dict]:
        q_vec = embed_query(query)
        scores = self.embeddings @ q_vec

        if category_filter:
            mask = self.meta["category"].isin(category_filter).values
            scores[~mask] = -np.inf

        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == -np.inf:
                continue
            row = self.meta.iloc[idx]
            tag = row["tag"]
            entry = {
                "rank": rank + 1,
                "tag": tag,
                "category": row["category"],
                "score": float(scores[idx]),
                "wiki_text": row.get("body_clean", ""),
                "copyrights": [],
            }
            if expand_copyright and row["category"] == "character":
                entry["copyrights"] = self.char_copyright.get(tag, [])
            results.append(entry)

        return results

    def search_by_tag(self, tag: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        idx = self.tag_to_idx.get(tag)
        if idx is None:
            raise KeyError(f"Tag '{tag}' not found in index")

        q_vec = self.embeddings[idx]
        scores = self.embeddings @ q_vec
        scores[idx] = -np.inf

        top_indices = np.argpartition(-scores, min(top_k + 1, len(scores) - 1))[: top_k + 1]
        top_indices = top_indices[np.argsort(-scores[top_indices])][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            row = self.meta.iloc[idx]
            entry = {
                "rank": rank + 1,
                "tag": row["tag"],
                "category": row["category"],
                "score": float(scores[idx]),
                "wiki_text": row.get("body_clean", ""),
                "copyrights": self.char_copyright.get(row["tag"], []) if row["category"] == "character" else [],
            }
            results.append(entry)
        return results
