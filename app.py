import os
from typing import List, Dict, Optional

import streamlit as st

from src.retriever import MultiDocRetriever, load_or_build_index
from src.llm_api import generate_llm_response, AVAILABLE_MODELS, DEFAULT_MODEL


DATA_DIR = "data"  # folder that contains the PDF syllabus, papers, etc.


def init_session_state() -> None:
    """Initialize chat history and cached retriever in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None


def ensure_retriever(force_rebuild: bool = False) -> MultiDocRetriever:
    """
    Lazily load or rebuild the document index and return a MultiDocRetriever.
    This is called whenever we need to run a RAG query.
    """
    if st.session_state.get("retriever") is not None and not force_rebuild:
        return st.session_state["retriever"]

    with st.spinner("Building / loading document index..."):
        retriever = load_or_build_index(DATA_DIR, force_rebuild=force_rebuild)
    st.session_state["retriever"] = retriever
    return retriever


def format_context(chunks: List[Dict]) -> str:
    """
    Turn retrieved chunks into a single context string for the LLM.
    Each chunk is expected to be a dict with keys:
    - text
    - source
    - page
    - score
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        page = c.get("page", "?")
        score = c.get("score", 0.0)
        lines.append(
            f"[{i}] (score={score:.3f}) Source: {os.path.basename(source)}, page {page}\n{c.get('text','').strip()}"
        )
    return "\n\n---\n\n".join(lines)


def render_sidebar() -> Dict:
    """Render the sidebar controls and return the configuration dict."""
    st.sidebar.header("⚙️ Settings")

    # Model selection
    model_display_names = list(AVAILABLE_MODELS.keys())
    default_index = model_display_names.index("LLaMA 3.1 8B (fast)") if "LLaMA 3.1 8B (fast)" in model_display_names else 0
    model_name_ui = st.sidebar.selectbox(
        "Model",
        model_display_names,
        index=default_index,
        help="Back-end LLM served via Groq API.",
    )
    model_name = AVAILABLE_MODELS[model_name_ui]

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Higher values = more creative, lower values = more deterministic.",
    )

    top_k = st.sidebar.slider(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many chunks to retrieve from the index in RAG mode.",
    )

    mode = st.sidebar.radio(
        "Answering mode",
        options=["RAG (with documents)", "Baseline (no documents)"],
        index=0,
    )

    force_rebuild = st.sidebar.checkbox(
        "Force rebuild index",
        value=False,
        help="If checked, the PDF corpus will be re-parsed and a fresh FAISS index will be created on the next question.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 About")
    st.sidebar.markdown(
        """
        This app is a **Multi-Document RAG assistant** for the course project.

        - Upload / store multiple PDFs in the `data/` folder.
        - The retriever chunks the text, builds dense embeddings (MiniLM-L6-v2),
          and indexes them using **FAISS**.
        - In **RAG mode**, the model only sees the retrieved chunks as context.
        - In **Baseline mode**, the model answers without access to the documents.
        """
    )

    return {
        "model_name": model_name,
        "temperature": temperature,
        "top_k": top_k,
        "mode": mode,
        "force_rebuild": force_rebuild,
    }


def main() -> None:
    st.set_page_config(
        page_title="Multi-Document RAG Assistant",
        page_icon="📄",
        layout="wide",
    )

    init_session_state()
    config = render_sidebar()

    st.title("📄 Multi-Document RAG Assistant")
    st.caption(
        "Ask questions about the uploaded PDFs. Toggle between **RAG** and **Baseline** "
        "to compare the impact of retrieval."
    )

    # Chat history display
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about the documents")
    if not user_query:
        return

    # Log user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            if config["mode"].startswith("RAG"):
                retriever = ensure_retriever(force_rebuild=config["force_rebuild"])
                retrieved_chunks = retriever.retrieve(user_query, top_k=config["top_k"])
                context_str = format_context(retrieved_chunks)

                with st.expander("🔍 Retrieved context", expanded=False):
                    st.write(context_str)

                answer = generate_llm_response(
                    question=user_query,
                    context=context_str,
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                )
            else:
                # Baseline: no documents, pure parametric answer
                answer = generate_llm_response(
                    question=user_query,
                    context=None,
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                )

            placeholder.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as exc:  # noqa: BLE001
            error_msg = f"⚠️ Failed to load retriever or model: {exc}"
            placeholder.error(error_msg)


if __name__ == "__main__":
    main()

