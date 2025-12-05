# evaluation/eval_pipeline.py

import sys
from pathlib import Path

# Add project root to sys.path so that `src` can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Any, Tuple

from src.llm_api import generate_llm_response, DEFAULT_MODEL
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
    - Build a context string (same style as app.py 里的 format_context).
    - Call the shared LLM API to generate an answer.

    Returns:
        answer: generated answer text
        retrieved_texts: list of plain-text chunk contents
    """

    chunks = retriever.retrieve(question, top_k=k)

    context_blocks: List[str] = []
    retrieved_texts: List[str] = []

    for i, c in enumerate(chunks, start=1):
        text = chunk_to_text(c)
        retrieved_texts.append(text)

        if isinstance(c, dict):
            source = c.get("source", "unknown")
            page = c.get("page", "?")
            score = c.get("score", 0.0)

            try:
                source_name = Path(source).name
            except Exception:
                source_name = str(source)

            header = f"[{i}] (score={score:.3f}) Source: {source_name}, page {page}"
            block = header + "\n" + text.strip()
        else:
            block = f"[{i}]\n{text.strip()}"

        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    answer = generate_llm_response(
        question=question,
        context=context_str,
        model_name=DEFAULT_MODEL,  
        temperature=0.2,
        max_tokens=256,
    )

    return answer, retrieved_texts


