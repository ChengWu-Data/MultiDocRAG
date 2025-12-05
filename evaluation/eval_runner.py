from evaluation.load_documents import build_eval_index
from evaluation.qa_loader import load_qa
from rag_pipeline import answer_question  

def run_evaluation():
    index, chunks = build_eval_index("evaluation/pdfs/")
    qa = load_qa("evaluation/qa_set.json")

    results = []
    for item in qa["answerable"]:
        q = item["question"]
        pred = answer_question(q, index, chunks, mode="rag")
        results.append({...})
