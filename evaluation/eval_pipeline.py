# evaluation/eval_pipeline.py

from sentence_transformers import SentenceTransformer
import numpy as np
from utils.indexer import search_faiss
from src.llm_api import generate_llm_response

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def eval_rag_answer(question, index, chunks, k=6):
    # encode question
    q_emb = embed_model.encode([question])
    scores, ids = search_faiss(index, q_emb, k)

    retrieved = [chunks[i] for i in ids]

    # build context
    context = "\n\n".join(retrieved)

    prompt = f"""
Use ONLY the provided context to answer the question.
If the context is insufficient, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}
"""
    answer = generate_llm_response(prompt=prompt, temperature=0.2, max_tokens=256)

    return answer, retrieved
