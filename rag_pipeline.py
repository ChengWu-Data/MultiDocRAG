# rag_pipeline.py (API version — replacing local HF model generation)

import argparse
import os

from src.retriever import MultiDocRetriever
from src.llm_api import generate_llm_response


# -----------------------------
# Paths and retriever
# -----------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")


def load_retriever(index_dir: str) -> MultiDocRetriever:
    retriever = MultiDocRetriever(
        model_name="all-MiniLM-L6-v2",
        max_chars=800,
        overlap_chars=150,
    )
    retriever.load(index_dir)
    return retriever


# -----------------------------
# Prompt builders
# -----------------------------

def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are a teaching assistant for a graduate-level financial economics class.

Use ONLY the provided context to answer the question.
If the context is insufficient, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}

Provide a concise answer in 1–2 short paragraphs.
"""


def build_baseline_prompt(question: str) -> str:
    return (
        "You are a general-purpose assistant.\n"
        "Answer the question below as best as you can.\n"
        "Do not assume you have access to any specific research papers.\n\n"
        f"Question:\n{question}\n"
    )


# -----------------------------
# RAG logic (now using API)
# -----------------------------

def run_rag_query(
    retriever: MultiDocRetriever,
    question: str,
    mode: str = "rag",
    k: int = 6,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:

    mode = mode.lower()
    if mode not in {"rag", "baseline"}:
        raise ValueError(f"Unknown mode: {mode}. Expected 'rag' or 'baseline'.")

    # Retrieval mode
    if mode == "rag":
        chunks = retriever.retrieve(question, k=k)

        context_blocks = []
        for c in chunks:
            header = f"[{c['doc_id']} — chunk {c['chunk_id']}]"
            context_blocks.append(header + "\n" + c["text"])

        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = build_rag_prompt(question, context)

    # Baseline mode
    else:
        prompt = build_baseline_prompt(question)

    # API call
    answer = generate_llm_response(
        prompt=prompt,
        temperature=temperature,
        max_tokens=256,
    )
    return answer


# -----------------------------
# CLI Interface
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="MultiDocRAG CLI")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rag", choices=["rag", "baseline"])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()

    print(f"Loading retriever from: {INDEX_DIR}")
    retriever = load_retriever(INDEX_DIR)

    print(f"\nQuestion:\n{args.question}")
    print(f"\nMode: {args.mode} (k={args.k})\n")

    answer = run_rag_query(
        retriever=retriever,
        question=args.question,
        mode=args.mode,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()


