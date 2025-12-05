# evaluation/eval_pipeline.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Any, Tuple

from utils.embedder import embed_model
from utils.indexer import search_faiss
from src.llm_api import generate_llm_response


def chunk_to_text(chunk: Any) -> str:
    """
    Convert a chunk to plain text.
    """
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def eval_rag_answer(
    question: str,
    index: Any,
    chunks: List[Any],
    k: int = 6,
) -> Tuple[str, List[str]]:
    """
    Evaluation-only RAG pipeline:
    - Perform FAISS retrieval from the evaluation index.
    - Build context.
    - Call the LLM (API) to generate an answer.
    """

    # Encode question → 1D vector
    q_emb = embed_model.encode([question])[0]

    # FAISS search
    scores, ids = search_faiss(index, q_emb, k)

    # Convert retrieved chunks to text
    retrieved_texts = [
        chunk_to_text(chunks[i])
        for i in ids
    ]

    context = "\n\n".join(retrieved_texts) if retrieved_texts else "No context retrieved."

    # Build prompt
    prompt = f"""
Use ONLY the provided context to answer the question.
If the context is insufficient, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}
"""

    # LLM API call
    answer = generate_llm_response(
        prompt=prompt,
        temperature=0.2,
        max_tokens=256,
    )

    return answer, retrieved_texts

