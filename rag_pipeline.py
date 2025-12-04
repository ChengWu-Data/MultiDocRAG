import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.retriever import MultiDocRetriever


# Load Retriever
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")

retriever = MultiDocRetriever(
    model_name="all-MiniLM-L6-v2",
    max_chars=800,
    overlap_chars=150,
)
retriever.load(INDEX_DIR)


# Load Causal LLM
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # you can change this anytime
print(f"Loading LLM: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()


# Helper: Maximum context length
def get_max_context_len(model) -> int:
    cfg = model.config
    for name in [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
    ]:
        if hasattr(cfg, name) and getattr(cfg, name) is not None:
            return int(getattr(cfg, name))
    return 1024


# Safe generation wrapper
def generate_from_model(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:

    max_ctx = get_max_context_len(model)

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    # If input too long, trim left
    if input_len >= max_ctx:
        input_ids = input_ids[:, -(max_ctx - 1):]
        input_len = input_ids.shape[1]

    # Make sure new tokens fit
    max_new_tokens = min(max_new_tokens, max_ctx - input_len)

    input_ids = input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# RAG Logic
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
You are a teaching assistant for a graduate-level financial economics class.

Use ONLY the provided context to answer the question.
If the context does not contain enough information, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}

Provide a concise answer.
"""


def run_rag(question: str):
    chunks = retriever.retrieve(question, k=6)
    context = "\n\n".join([f"[{c['doc_id']}|{c['chunk_id']}] {c['text']}" for c in chunks])
    prompt = build_rag_prompt(question, context)
    answer = generate_from_model(prompt)
    return answer


# CLI Interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    print("\n🔍 Running RAG Query...\n")
    print("Question:", args.question)
    print("\nAnswer:\n")
    print(run_rag(args.question))
