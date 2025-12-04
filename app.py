import os
import shutil
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.retriever import MultiDocRetriever


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded_pdfs")


st.set_page_config(page_title="MultiDocRAG Demo", layout="wide")
st.title("MultiDocRAG: Demo with PDF Upload")

st.markdown(
    """
This is a demo for the MultiDocRAG pipeline.

It uses:
- a FAISS index in `index_store/`
- a local open-source language model
- retrieval-augmented prompting to answer questions grounded in the documents
"""
)

# Sidebar settings


with st.sidebar:
    st.header("Settings")

    default_model_name = "sshleifer/tiny-gpt2"
    model_name = st.text_input("Model name", value=default_model_name)

    k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=6, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

    mode = st.radio(
        "Answering mode",
        options=["RAG", "Baseline"],
        index=0,
        help="Baseline: LLM without retrieved context. RAG: LLM with retrieved context.",
    )



# Cached loaders

@st.cache_resource(show_spinner=True)
def load_retriever_and_index(index_dir: str) -> MultiDocRetriever:
    retriever = MultiDocRetriever(
        model_name="all-MiniLM-L6-v2",
        max_chars=800,
        overlap_chars=150,
    )
    retriever.load(index_dir)
    return retriever


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer_cached(model_name: str):
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
    Safe generation wrapper compatible with GPT-2 style and larger causal LMs.
    """

    max_ctx = get_max_context_len(model)

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

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



# PDF upload + index rebuild

st.subheader("1. Upload PDFs and rebuild index")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files to rebuild the index.",
    type=["pdf"],
    accept_multiple_files=True,
)

col_build, col_status = st.columns([1, 2])

with col_build:
    rebuild_clicked = st.button("Build / Rebuild index from uploaded PDFs")

with col_status:
    index_exists = (
        os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy"))
        and os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))
        and os.path.exists(os.path.join(INDEX_DIR, "chunks.json"))
        and os.path.exists(os.path.join(INDEX_DIR, "meta.json"))
    )
    if index_exists:
        st.success("Existing index detected in `index_store/`.")
    else:
        st.warning("No existing index found in `index_store/`. Please upload PDFs and build an index.")

if rebuild_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before rebuilding the index.")
    else:
        with st.spinner("Building index from uploaded PDFs..."):
            # Clear previous uploads and recreate directory
            if os.path.isdir(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            # Save uploaded PDFs
            pdf_paths = []
            for f in uploaded_files:
                out_path = os.path.join(UPLOAD_DIR, f.name)
                with open(out_path, "wb") as out_f:
                    out_f.write(f.read())
                pdf_paths.append(out_path)

            # Build a new retriever and index
            new_retriever = MultiDocRetriever(
                model_name="all-MiniLM-L6-v2",
                max_chars=800,
                overlap_chars=150,
            )
            for pdf_path in pdf_paths:
                new_retriever.add_pdf(pdf_path)

            new_retriever.build_index(show_progress=True)
            os.makedirs(INDEX_DIR, exist_ok=True)
            new_retriever.save(INDEX_DIR)

            # Clear cached retriever and reload
            load_retriever_and_index.clear()
            st.success("Index rebuilt successfully from uploaded PDFs.")

# ==========================
# Load retriever and model
# ==========================

try:
    retriever = load_retriever_and_index(INDEX_DIR)
    tokenizer, model = load_model_and_tokenizer_cached(model_name)
    st.success("Retriever and model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load retriever or model: {e}")
    retriever = None
    model = None

# Question answering

st.subheader("2. Ask questions")

question = st.text_area("Enter your question", height=120)

if st.button("Run"):
    if retriever is None or model is None:
        st.error("Retriever or model not available. Make sure the index is built and the model is loaded.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        if mode == "RAG":
            chunks = retriever.retrieve(question, k=k)
            context_blocks = []
            for c in chunks:
                header = f"[{c['doc_id']} — chunk {c['chunk_id']}]"
                context_blocks.append(header + "\n" + c["text"])
            context = "\n\n".join(context_blocks)
            prompt = build_rag_prompt(question, context)
        else:
            context = None
            prompt = (
                "You are a general-purpose assistant.\n"
                "Answer the question below as best as you can.\n"
                "Do not assume you have access to any specific research papers.\n\n"
                f"Question:\n{question}\n"
            )

        with st.spinner("Generating answer..."):
            answer = generate_from_model(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
            )

        st.subheader("Answer")
        st.write(answer)

        if mode == "RAG" and context is not None:
            st.subheader("Retrieved Context")
            with st.expander("Show retrieved chunks"):
                st.text(context)










































































































































