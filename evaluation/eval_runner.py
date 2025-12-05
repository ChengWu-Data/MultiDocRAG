from evaluation.load_documents import build_eval_index
from evaluation.eval_pipeline import eval_rag_answer
from evaluation.qa_loader import load_qa

def run_evaluation():
    index, chunks = build_eval_index("evaluation/pdfs/")
    qa = load_qa("evaluation/qa_set.json")

    results = []
    for item in qa["answerable"]:
        q = item["question"]
        ans, retrieved = eval_rag_answer(q, index, chunks, k=6)
        results.append({
            "id": item["id"],
            "question": q,
            "gold": item["gold_answer"],
            "answer": ans,
            "chunks": retrieved,
        })
