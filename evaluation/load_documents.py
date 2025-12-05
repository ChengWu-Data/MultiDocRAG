import os
from utils.loader import extract_text  
from utils.chunker import chunk_text
from utils.embedder import embed_model  
from utils.indexer import build_faiss_index

def build_eval_index(pdf_dir):
    all_chunks = []
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, fname)
            text = extract_text(pdf_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    embeddings = embed_model.encode(all_chunks)
    index = build_faiss_index(embeddings)
    return index, all_chunks
