import os
import shutil
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.retriever import MultiDocRetriever


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded_pdfs")

EXAMPLE_QUESTIONS = [
    "Summarize the main ideas discussed across these documents.",
    "What are the main sources of risk mentioned across the documents?",
    "Compare how different documents describe the same concept or methodology.",
    "What are the key assumptions and limitations highlighted in these documents?",
    "How do the documents differ in their conclusions or policy implications?",
]
EXAMPLE_LABEL = "(Choose an example question)"

st.set_page_config(page_title="MultiDocRAG Demo", layout="wide")
st.title("MultiDocRAG: Multi-Document RAG Demo")

st.markdown(
    """
This app demonstrates the MultiDocRAG pipeline:

- uses a FAISS index stored in `index_store/`
- loads a local open-source language model
- answers questions either with or without retrieved context (Baseline vs RAG)
"""
)

# --- Human-facing explanation about the model choice ---

st.markdown(
    """
### Why a small default model?

By default, we load a very small toy model, `sshleifer/tiny-gpt2`.  
It runs quickly on a laptop CPU, but the answers can be simple or even a bit noisy.  
The primary goal of this demo is to showcase the **pipeline**:

> PDFs → chunks → embeddings → retrieval → prompt construction → LLM answer

rather than raw model power.

For stronger answers, you can replace the model in the sidebar with a small
instruction-tuned chat model such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
when running on CPU/GPU.  
The exact same retrieval pipeline also works with larger models (e.g., LLaMA/Mistral/Phi)
that we use in our experiments and evaluation.
"""
)

st.markdown("---")


# ==========================
# Sidebar settings
# ==========================

with st.sidebar:
    st.header("Settings")

    default_model_name = "sshleifer/tiny-gpt2"
    model_name = st.text_input("Model name", value=default_model_name)
    st.caption(
        "This should be a Hugging Face causal LM name.\n\n"
        "For better answers, try an instruction-tuned chat model such as "
        "`TinyLlama/TinyLlama-1.1B-Chat-v1.0`."
    )

    k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=6, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

    mode = st.radio(
        "Answering mode",
        options=["RAG", "Baseline"],
        index=0,
        help="Baseline: LLM without retrieved context. RAG: LLM with retrieved context.",
    )


# ==========================
# Cached loaders
# ==========================

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

    # ensure pad token exists
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
    """Safe generation wrapper — returns ONLY the newly generated continuation text."""
    max_ctx = get_max_context_len(model)

    # Encode prompt
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    # Truncate if prompt is too long
    if input_len >= max_ctx:
        keep_len = max_ctx - 1
        keep_len = max(1, keep_len)
        input_ids = input_ids[:, -keep_len:]
        input_len = keep_len

    # How many new tokens we are allowed to generate
    available_for_gen = max_ctx - input_len
    max_new_tokens = max(1, min(max_new_tokens, available_for_gen))

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

    # Only decode the NEW tokens, not the whole prompt + completion
    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text.strip()


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


# ==========================
# Section 1: Upload PDFs & rebuild index
# ==========================

st.subheader("1. Upload PDFs and rebuild index")
st.caption("Upload one or more PDFs, then click the button to (re)build a fresh FAISS index from them.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files to rebuild the FAISS index.",
    type=["pdf"],
    accept_multiple_files=True,
)

col_build, col_status = st.columns([1, 2])

with col_build:
    rebuild_clicked = st.button("Build / Rebuild index from uploaded PDFs", use_container_width=True)

with col_status:
    index_exists = (
        os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy"))
        and os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))
        and os.path.exists(os.path.join(INDEX_DIR, "chunks.json"))
        and os.path.exists(os.path.join(INDEX_DIR, "meta.json"))
    )
    if index_exists:
        st.success("Existing index detected in `index_store/`.")
        # Try to show a short summary of indexed documents (if meta.json has that info)
        meta_path = os.path.join(INDEX_DIR, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                docs = meta.get("documents") or meta.get("docs") or []
                if isinstance(docs, list) and docs:
                    doc_names = []
                    for d in docs:
                        if isinstance(d, dict):
                            name = d.get("doc_id") or d.get("name") or d.get("filename")
                        else:
                            name = str(d)
                        if name:
                            doc_names.append(name)
                    if doc_names:
                        joined = ", ".join(doc_names)
                        if len(joined) > 200:
                            joined = joined[:200] + "..."
                        st.caption(f"Indexed documents: {joined}")
        except Exception:
            # If meta format is unexpected, just skip showing details
            pass
    else:
        st.warning("No index found in `index_store/`. Please upload PDFs and build an index.")

if rebuild_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before rebuilding the index.")
    else:
        with st.spinner("Building index from uploaded PDFs..."):
            # clear previous uploaded_pdfs directory
            if os.path.isdir(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            pdf_paths = []
            for f in uploaded_files:
                out_path = os.path.join(UPLOAD_DIR, f.name)
                with open(out_path, "wb") as out_f:
                    out_f.write(f.read())
                pdf_paths.append(out_path)

            # build a new retriever from these PDFs
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

            # clear cached retriever so it reloads with the new index
            load_retriever_and_index.clear()
            st.success("Index rebuilt successfully from uploaded PDFs.")

st.markdown("---")


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


# ==========================
# Callback: load example question
# ==========================

def load_example_callback():
    choice = st.session_state.get("example_choice", EXAMPLE_LABEL)
    if choice != EXAMPLE_LABEL:
        st.session_state["question_text"] = choice


# ==========================
# Section 2: Question answering
# ==========================

st.subheader("2. Ask a question over the document collection")
st.caption(
    "RAG mode uses retrieved chunks from your PDFs; "
    "Baseline mode ignores them and lets the model answer from its own knowledge."
)

# 初始化问题文本的 session_state
if "question_text" not in st.session_state:
    st.session_state["question_text"] = ""

# 文本输入框绑定到 question_text
question = st.text_area(
    "Enter your question",
    height=120,
    key="question_text",
)

col_left, col_right = st.columns([2, 1])

with col_left:
    run_clicked = st.button("Run", type="primary", use_container_width=True)

with col_right:
    st.selectbox(
        "Example question",
        [EXAMPLE_LABEL] + EXAMPLE_QUESTIONS,
        index=0,
        key="example_choice",
    )
    st.button("Load example", on_click=load_example_callback, use_container_width=True)

if run_clicked:
    # 每次 Run 时，从 session_state 里拿当前问题文本
    question = st.session_state.get("question_text", "")

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
            chunks = []
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

        col_ans, col_ctx = st.columns([2, 1])

        with col_ans:
            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show full prompt used"):
                st.code(prompt)

        with col_ctx:
            if mode == "RAG" and chunks:
                st.subheader("Top retrieved chunks")
                for i, c in enumerate(chunks, start=1):
                    with st.expander(f"Chunk {i} — {c.get('doc_id', 'doc')} [#{c.get('chunk_id', '?')}]"):
                        st.write(c.get("text", ""))
                        score = c.get("score", None)
                        if score is not None:
                            st.caption(f"Score: {score:.4f}")
            elif mode == "Baseline":
                st.info("Baseline mode: no retrieved context is used.")
