# evaluation/eval_pipeline.py

import sys
from pathlib import Path

# Add project root to sys.path so that `src` can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Any, Tuple

from src.llm_api import generate_llm_response
from src.retriever import MultiDocRetriever


def chunk_to_text(chunk: Any) -> str:
    """
    Convert a retrieved chunk object to plain text.
    """
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def eval_rag_answer(
    question: str,
    retriever: MultiDocRetriever,
    k: int = 6,
) -> Tuple[str, List[str]]:
    """
    Evaluation-only RAG pipeline:
    - Use MultiDocRetriever to retrieve top-k chunks.
    - Build a context string.
    - Call the LLM API to generate an answer.

    Returns:
        answer: generated answer text
        retrieved_texts: list of plain-text chunk contents
    """

    chunks = retriever.retrieve(question, k=k)

    context_blocks: List[str] = []
    retrieved_texts: List[str] = []

    for c in chunks:
        text = chunk_to_text(c)
        retrieved_texts.append(text)

        if isinstance(c, dict):
            header = f"[{c.get('doc_id', 'doc')} — chunk {c.get('chunk_id', '?')}]"
            context_blocks.append(header + "\n" + text)
        else:
            context_blocks.append(text)

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    prompt = f"""
Use ONLY the provided context to answer the question.
If the context is insufficient, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}
"""

    answer = generate_llm_response(
        prompt=prompt,
        temperature=0.2,
        max_tokens=256,
    )

    return answer, retrieved_texts


