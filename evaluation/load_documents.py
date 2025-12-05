# evaluation/load_documents.py

import os
from typing import List, Tuple, Any

from utils.loader import extract_text
from utils.chunker import chunk_text
from utils.embedder import embed_model
from utils.indexer import build_faiss_index


def build_eval_index(pdf_dir: str) -> Tuple[Any, List[Any]]:
    """
    Build an evaluation-only FAISS index using the PDFs located in evaluation/pdfs/.
    This does NOT modify the main system index_store.
    """

    all_chunks: List[Any] = []

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, fname)
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No chunks found. Ensure PDFs exist in: {pdf_dir}")

    # Extract raw text for embedding
    texts = [
        c["text"] if isinstance(c, dict) else str(c)
        for c in all_chunks
    ]

    embeddings = embed_model.encode(texts)
    index = build_faiss_index(embeddings)

    return index, all_chunks
