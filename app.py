import os
import shutil
import json
import streamlit as st

from src.retriever import MultiDocRetriever
from src.llm_api import generate_llm_response  # 🔹 使用 API LLM，而不是本地模型

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

# ==========================
# Global styling & hero header
# ==========================

st.markdown(
    """
    <style>
    /* Purple aesthetic full background */
    .main {
        background: linear-gradient(180deg, #ede9fe 0%, #f5f3ff 40%, #faf5ff 100%) !important;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .app-hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #4c1d95;
    }
    .app-hero-subtitle {
        font-size: 0.98rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .app-hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #4c1d95;
        background: rgba(129, 140, 248, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.35);
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .app-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.9rem;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(76, 29, 149, 0.05);
        margin-bottom: 1.5rem;
    }
    .app-card h3, .app-card h2 {
        margin-top: 0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3e8ff 0%, #f8f5ff 60%, #ede9fe 100%);
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4c1d95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
        <div class="app-hero-tag">
            <span>Multi-document QA</span>
            <span>·</span>
            <span>RAG + lightweight memory</span>
        </div>
        <div class="app-hero-title">MultiDocRAG: Multi-Document RAG Demo</div>
        <div class="app-hero-subtitle">
            Ask questions across multiple research papers using a retrieval-augmented pipeline
            with an API-hosted language model and session-level conversational memory.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
This app demonstrates the MultiDocRAG pipeline:

- uses a FAISS index stored in `index_store/`
- retrieves relevant chunks from uploaded PDFs
- builds prompts with recent conversation history
- calls an external LLM API (configured in `src/llm_api.py`) to generate answers
"""
)

st.markdown("---")

# ==========================
# Sidebar settings
# ==========================

with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown(
        "LLM is served via an external API (see `src/llm_api.py`). "
        "You can adjust retrieval and decoding settings below."
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
# Cached loader for retriever
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

# ==========================
# Prompt builders with conversation memory
# ==========================

def format_history(history, max_turns: int = 3) -> str:
    """
    Format recent conversation history for injection into the prompt.
    max_turns = number of user–assistant exchanges to keep.
    """
    if not history:
        return "No previous conversation.\n"
    recent = history[-2 * max_turns :]
    lines = []
    for role, text in recent:
        lines.append(f"{role}: {text}")
    return "\n".join(lines) + "\n"


def build_rag_prompt(question: str, context: str, history) -> str:
    history_text = format_history(history)
    return f"""You are a teaching assistant for a graduate-level financial economics class.

Here is the recent conversation with the user:
{history_text}
Use ONLY the provided context from the documents to answer the new question.
If the context is insufficient, say:
"The context does not provide enough information to answer fully."

Context:
{context}

Question:
{question}

Provide a concise answer in 1–2 short paragraphs.
"""


def build_baseline_prompt(question: str, history) -> str:
    history_text = format_history(history)
    return (
        "You are a general-purpose assistant.\n"
        "Here is the recent conversation with the user:\n"
        f"{history_text}\n"
        "Answer the new question below as best as you can.\n"
        "Do not assume you have access to any specific research papers.\n\n"
        f"Question:\n{question}\n"
    )

# ==========================
# Section 1: Upload PDFs & rebuild index
# ==========================

st.markdown('<div class="app-card">', unsafe_allow_html=True)

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
            pass
    else:
        st.warning("No index found in `index_store/`. Please upload PDFs and build an index.")

if rebuild_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before rebuilding the index.")
    else:
        with st.spinner("Building index from uploaded PDFs..."):
            if os.path.isdir(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            pdf_paths = []
            for f in uploaded_files:
                out_path = os.path.join(UPLOAD_DIR, f.name)
                with open(out_path, "wb") as out_f:
                    out_f.write(f.read())
                pdf_paths.append(out_path)

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

            load_retriever_and_index.clear()
            st.success("Index rebuilt successfully from uploaded PDFs.")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# Load retriever
# ==========================

try:
    retriever = load_retriever_and_index(INDEX_DIR)
    st.success("Retriever loaded successfully. LLM is provided via API (see `src/llm_api.py`).")
except Exception as e:
    st.error(f"Failed to load retriever: {e}")
    retriever = None

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

st.markdown('<div class="app-card">', unsafe_allow_html=True)

st.subheader("2. Ask a question over the document collection")
st.caption(
    "RAG mode uses retrieved chunks from your PDFs; "
    "Baseline mode ignores them and lets the model answer from its own knowledge."
)

# Initialize session states
if "question_text" not in st.session_state:
    st.session_state["question_text"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = []

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

with st.expander("Conversation history in this session"):
    if st.session_state["history"]:
        for role, text in st.session_state["history"]:
            st.markdown(f"**{role}:** {text}")
    else:
        st.caption("No conversation history yet. Ask a question to start the dialogue.")

if run_clicked:
    question = st.session_state.get("question_text", "")

    if retriever is None:
        st.error("Retriever not available. Make sure the index is built successfully.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Append user question
        st.session_state["history"].append(("user", question))
        history = st.session_state["history"]

        if mode == "RAG":
            chunks = retriever.retrieve(question, k=k)
            context_blocks = []
            for c in chunks:
                header = f"[{c['doc_id']} — chunk {c['chunk_id']}]"
                context_blocks.append(header + "\n" + c["text"])
            context = "\n\n".join(context_blocks)
            prompt = build_rag_prompt(question, context, history)
        else:
            chunks = []
            context = None
            prompt = build_baseline_prompt(question, history)

        with st.spinner("Generating answer via API..."):
            try:
                answer = generate_llm_response(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=256,
                )
            except Exception as e:
                st.error(f"LLM API call failed: {e}")
                answer = None

        if answer is not None:
            st.session_state["history"].append(("assistant", answer))

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
                        with st.expander(
                            f"Chunk {i} — {c.get('doc_id', 'doc')} [#{c.get('chunk_id', '?')}]"
                        ):
                            st.write(c.get("text", ""))
                            score = c.get("score", None)
                            if score is not None:
                                st.caption(f"Score: {score:.4f}")
                elif mode == "Baseline":
                    st.info("Baseline mode: no retrieved context is used.")

st.markdown("</div>", unsafe_allow_html=True)
