# evaluation/load_documents.py

import sys
from pathlib import Path

# Add project root to sys.path so that `src` can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from typing import List
from src.retriever import MultiDocRetriever


def build_eval_retriever(pdf_dir: str) -> MultiDocRetriever:
    """
    Build an in-memory MultiDocRetriever using PDFs under evaluation/pdfs/.
    This does NOT touch index_store/ or the main app index.
    """

    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    retriever = MultiDocRetriever(
        chunk_size=800,
        chunk_overlap=150,
    )


    pdf_paths: List[Path] = [
        p for p in pdf_dir_path.iterdir()
        if p.suffix.lower() == ".pdf"
    ]

    if not pdf_paths:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    for p in pdf_paths:
        retriever.add_pdf(str(p))

    # Build index in memory, do not save to disk
    retriever.build_index()

    return retriever

