import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.retriever import MultiDocRetriever



# Paths and retriever setup

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


# LLM loading

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


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


def generate_from_model(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Safe, context-aware generation wrapper.
    Prevents context overflow and supports temperature / top-p sampling.
    """

    max_ctx = get_max_context_len(model)

    # Tokenize without truncation first
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    # If prompt is too long, keep only the last (max_ctx - 1) tokens
    if input_len >= max_ctx:
        keep_len = max_ctx - 1
        if keep_len <= 0:
            keep_len = 1
        input_ids = input_ids[:, -keep_len:]
        input_len = keep_len

    available_for_gen = max_ctx - input_len
    if available_for_gen <= 0:
        max_new_tokens = 1
    else:
        max_new_tokens = min(max_new_tokens, available_for_gen)

    device = model.device
    input_ids = input_ids.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True if (temperature > 0 or top_p < 1.0) else False,
        temperature=max(1e-5, float(temperature)),
        top_p=float(top_p),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# RAG logic

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


def run_rag_query(
    retriever: MultiDocRetriever,
    tokenizer,
    model,
    question: str,
    mode: str = "rag",
    k: int = 6,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    mode = mode.lower()
    if mode not in {"rag", "baseline"}:
        raise ValueError(f"Unknown mode: {mode}. Expected 'rag' or 'baseline'.")

    if mode == "rag":
        chunks = retriever.retrieve(question, k=k)
        context_blocks = []
        for c in chunks:
            header = f"[{c['doc_id']} — chunk {c['chunk_id']}]"
            context_blocks.append(header + "\n" + c["text"])
        context = "\n\n".join(context_blocks)
        prompt = build_rag_prompt(question, context)
    else:
        context = None
        prompt = build_baseline_prompt(question)

    answer = generate_from_model(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
    )
    return answer


# CLI interface

def main():
    parser = argparse.ArgumentParser(description="MultiDocRAG CLI")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the system.")
    parser.add_argument(
        "--mode",
        type=str,
        default="rag",
        choices=["rag", "baseline"],
        help="Answering mode: 'rag' uses retrieved context, 'baseline' does not.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of top retrieved chunks to use in RAG mode.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="Hugging Face model name for the causal LM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p.",
    )

    args = parser.parse_args()

    print(f"Loading retriever from: {INDEX_DIR}")
    retriever = load_retriever(INDEX_DIR)

    print(f"Loading model: {args.model_name}")
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    print("\nQuestion:")
    print(args.question)
    print(f"\nMode: {args.mode} (k={args.k})\n")

    answer = run_rag_query(
        retriever=retriever,
        tokenizer=tokenizer,
        model=model,
        question=args.question,
        mode=args.mode,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()

