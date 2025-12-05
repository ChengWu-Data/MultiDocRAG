"""
Retriever utilities for the Multi-Document RAG system.

Responsibilities:
- Parse PDFs under a given directory.
- Chunk text (800 characters, 150 overlap).
- Build dense embeddings with MiniLM-L6-v2.
- Build a FAISS index for fast vector search.
- (Optionally) combine with a BM25 lexical retriever.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:  # Try new pypdf first; fall back to PyPDF2 for compatibility
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    from PyPDF2 import PdfReader  # type: ignore

import faiss


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TextChunk:
    text: str
    source: str
    page: int
    chunk_id: int


def _slide_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based sliding window chunking."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:  # Just in case
            break
    return chunks


def read_pdf(path: str) -> List[str]:
    """Return a list of page texts from a PDF file."""
    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append(text)
    return pages


class MultiDocRetriever:
    """
    Hybrid dense + lexical retriever over multiple PDFs.

    Attributes:
    - model: sentence-transformers embedding model
    - text_chunks: list of TextChunk
    - embeddings: np.ndarray [num_chunks, dim]
    - bm25: BM25Okapi instance for lexical scores
    - index: FAISS index (inner-product / cosine similarity)
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        alpha: float = 0.7,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.alpha = alpha  # weight for dense vs lexical scores

        self.model = SentenceTransformer(self.embedding_model_name)
        self.text_chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    # -----------------------------
    # Building the index
    # -----------------------------
    def add_pdf(self, pdf_path: str) -> None:
        """Parse a PDF and append its chunks to `self.text_chunks`."""
        pages = read_pdf(pdf_path)
        chunk_id = len(self.text_chunks)
        for page_number, page_text in enumerate(pages, start=1):
            for chunk in _slide_window(page_text, self.chunk_size, self.chunk_overlap):
                self.text_chunks.append(
                    TextChunk(
                        text=chunk,
                        source=pdf_path,
                        page=page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1

    def add_directory(self, data_dir: str) -> None:
        """Add all PDF files under `data_dir`."""
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    full_path = os.path.join(root, fname)
                    self.add_pdf(full_path)

    def build_embeddings(self) -> None:
        """Encode all chunks with the embedding model."""
        if not self.text_chunks:
            raise ValueError("No text chunks to embed. Did you call add_pdf/add_directory?")
        texts = [c.text for c in self.text_chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.embeddings = embeddings

    def build_bm25(self) -> None:
        """Build BM25 index over the chunks."""
        if not self.text_chunks:
            raise ValueError("No text chunks for BM25.")
        tokenized = [c.text.lower().split() for c in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized)

    def build_faiss(self) -> None:
        """Create FAISS index over dense embeddings (cosine via inner-product)."""
        if self.embeddings is None:
            raise ValueError("Embeddings not built.")
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings.astype("float32"))
        self.index = index

    def build_index(self) -> None:
        """End-to-end index building: embeddings + FAISS + BM25."""
        self.build_embeddings()
        self.build_faiss()
        self.build_bm25()

    # -----------------------------
    # Saving / loading
    # -----------------------------
    def save_index(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)

        # Save embeddings
        if self.embeddings is None:
            raise ValueError("Cannot save before building embeddings/index.")
        np.save(os.path.join(index_dir, "embeddings.npy"), self.embeddings)

        # Save metadata
        meta = [asdict(c) for c in self.text_chunks]
        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Save FAISS index
        if self.index is None:
            raise ValueError("FAISS index is not built.")
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))

    def load_index(self, index_dir: str) -> None:
        """Load embeddings, metadata, and FAISS index from disk."""
        emb_path = os.path.join(index_dir, "embeddings.npy")
        chunks_path = os.path.join(index_dir, "chunks.json")
        faiss_path = os.path.join(index_dir, "faiss.index")

        if not (os.path.exists(emb_path) and os.path.exists(chunks_path) and os.path.exists(faiss_path)):
            raise FileNotFoundError(
                f"Missing saved index files under {index_dir}. "
                "Please rebuild the index from PDFs."
            )

        self.embeddings = np.load(emb_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.text_chunks = [TextChunk(**m) for m in meta]

        self.index = faiss.read_index(faiss_path)

        # Rebuild BM25 from loaded chunks
        tokenized = [c.text.lower().split() for c in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized)

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k chunks for the query.
        Returns a list of dicts with keys: text, source, page, score.
        """
        if self.index is None or self.embeddings is None or self.bm25 is None:
            raise ValueError("Index not built / loaded.")

        # Dense similarity
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        q_emb = q_emb.astype("float32")[None, :]
        scores_dense, indices = self.index.search(q_emb, top_k)
        dense_scores = scores_dense[0]
        faiss_indices = indices[0]

        # BM25 lexical scores (convert to same scale)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        # Normalize BM25 using min-max to [0,1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_norm = bm25_scores

        results: List[Dict] = []
        for rank, idx in enumerate(faiss_indices):
            if idx < 0:
                continue
            chunk = self.text_chunks[int(idx)]
            dense_score = float(dense_scores[rank])
            bm25_score = float(bm25_norm[int(idx)])
            combined = self.alpha * dense_score + (1.0 - self.alpha) * bm25_score

            results.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": combined,
                }
            )

        # Sort again by combined score (descending) to be safe
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def load_or_build_index(
    data_dir: str,
    index_dir: str = "indexes/default",
    force_rebuild: bool = False,
) -> MultiDocRetriever:
    """
    Load an existing index from `index_dir` if available; otherwise build from PDFs
    under `data_dir`. If `force_rebuild` is True, always rebuild from scratch.
    """
    retriever = MultiDocRetriever()

    if not force_rebuild:
        try:
            retriever.load_index(index_dir)
            return retriever
        except FileNotFoundError:
            # Fall back to building below
            pass

    # Build from PDFs
    retriever.add_directory(data_dir)
    retriever.build_index()
    retriever.save_index(index_dir)
    return retriever




