# evaluation/eval_runner.py

import sys
from pathlib import Path

# Add project root to sys.path so that `src` and `evaluation` can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
from typing import List, Dict, Any

from evaluation.load_documents import build_eval_retriever
from evaluation.eval_pipeline import eval_rag_answer
from evaluation.qa_loader import load_qa


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
QA_PATH = BASE_DIR / "qa_set.json"
RESULTS_DIR = BASE_DIR / "results"
OUT_CSV = RESULTS_DIR / "eval_outputs.csv"


def run_evaluation() -> None:
    """
    Fully automated evaluation pipeline:
    - Build an in-memory retriever from evaluation/pdfs/
    - Load QA dataset from evaluation/qa_set.json
    - Run RAG answering for answerable & unanswerable questions
    - Save results to evaluation/results/eval_outputs.csv
    """

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {PDF_DIR}")

    retriever = build_eval_retriever(str(PDF_DIR))

    qa = load_qa(QA_PATH)
    answerable: List[Dict[str, Any]] = qa.get("answerable", [])
    unanswerable: List[Dict[str, Any]] = qa.get("unanswerable", [])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "group",
        "mode",
        "is_unanswerable",
        "question",
        "gold_answer",
        "system_answer",
        "retrieved_chunks",
    ]

    rows: List[Dict[str, Any]] = []

    # Answerable questions (mode = RAG)
    for item in answerable:
        qid = item.get("id", "")
        question = item["question"]
        gold = item.get("gold_answer", "")
        group = item.get("group", "answerable")

        ans, retrieved_texts = eval_rag_answer(
            question=question,
            retriever=retriever,
            k=6,
        )

        rows.append(
            {
                "id": qid,
                "group": group,
                "mode": "rag",
                "is_unanswerable": 0,
                "question": question,
                "gold_answer": gold,
                "system_answer": ans,
                "retrieved_chunks": " ||| ".join(retrieved_texts),
            }
        )

    # Unanswerable questions
    for item in unanswerable:
        qid = item.get("id", "")
        question = item["question"]
        group = item.get("group", "unanswerable")

        ans, retrieved_texts = eval_rag_answer(
            question=question,
            retriever=retriever,
            k=6,
        )

        rows.append(
            {
                "id": qid,
                "group": group,
                "mode": "rag",
                "is_unanswerable": 1,
                "question": question,
                "gold_answer": "",
                "system_answer": ans,
                "retrieved_chunks": " ||| ".join(retrieved_texts),
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Evaluation completed. Results saved to: {OUT_CSV}")


if __name__ == "__main__":
    run_evaluation()

