"""
retriever.py

Document ingestion and retrieval module for MultiDocRAG.
Implements PDF loading, cleaning, chunking, embeddings, 
and FAISS-based vector search.
"""

import os
import json
from typing import List, Dict, Optional

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import torch


class MultiDocRetriever:
    """
    Multi-document retriever:
    - PDF ingestion
    - cleaning + chunking
    - sentence-transformer embeddings
    - FAISS vector search
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_chars: int = 800,
        overlap_chars: int = 150,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

        self.chunks: List[str] = []
        self.meta: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None

    
    # PDF LOADING / CLEANING / CHUNKING
    
    @staticmethod
    def _load_pdf_text(pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(page_text)
        return "\n".join(texts)

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
        return cleaned.strip()

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = end - self.overlap_chars

        return chunks

    # INGESTION
    
    def add_pdf(self, pdf_path: str, doc_id: Optional[str] = None):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if doc_id is None:
            doc_id = os.path.basename(pdf_path)

        raw_text = self._load_pdf_text(pdf_path)
        cleaned = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned)

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.meta.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "global_id": start_idx + i,
                }
            )

        print(f"[INFO] Ingested {doc_id}: {len(chunks)} chunks.")

    def add_pdfs_from_dir(self, pdf_dir: str, recursive: bool = False):
        if not os.path.isdir(pdf_dir):
            raise NotADirectoryError(f"Directory not found: {pdf_dir}")

        count = 0
        for root, dirs, files in os.walk(pdf_dir):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    self.add_pdf(os.path.join(root, fname))
                    count += 1
            if not recursive:
                break

        print(f"[INFO] Finished ingesting PDFs. Total files: {count}")

    
    # EMBEDDINGS + INDEX
    
    def build_index(self, show_progress: bool = True):
        if not self.chunks:
            raise ValueError("No chunks to index. Ingest PDFs first.")

        print(f"[INFO] Computing embeddings for {len(self.chunks)} chunks...")

        self.embeddings = self.model.encode(
            self.chunks,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            device=self.device,
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print(f"[INFO] FAISS index built. dim={dim}, size={self.index.ntotal}")

    # RETRIEVAL
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None or self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        q_emb = self.model.encode([query], convert_to_numpy=True, device=self.device)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            meta = self.meta[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(dist),
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "global_id": meta["global_id"],
                    "text": self.chunks[idx],
                }
            )
        return results

    
    # SAVE / LOAD
    
    def save(self, out_dir: str):
        if self.embeddings is None or self.index is None:
            raise ValueError("Nothing to save. Build index first.")

        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(out_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f)

        faiss.write_index(self.index, os.path.join(out_dir, "faiss.index"))
        print(f"[INFO] Saved index to {out_dir}")

    def load(self, in_dir: str):
        emb_path = os.path.join(in_dir, "embeddings.npy")
        chunks_path = os.path.join(in_dir, "chunks.json")
        meta_path = os.path.join(in_dir, "meta.json")
        index_path = os.path.join(in_dir, "faiss.index")

        if not all(os.path.exists(p) for p in [emb_path, chunks_path, meta_path, index_path]):
            raise FileNotFoundError("Missing saved index files.")

        self.embeddings = np.load(emb_path)

        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.index = faiss.read_index(index_path)
        print(f"[INFO] Loaded index from {in_dir}")
