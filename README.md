# **MultiDocRAG — Multi-Document Retrieval-Augmented Generation System**

**MultiDocRAG** is a full retrieval-augmented question answering system that allows users to upload multiple documents, build a vector index, and query content using a modern LLM with contextual grounding and lightweight conversational memory.

This project includes:

* Multi-PDF ingestion & FAISS indexing
* A complete RAG pipeline
* Session-level conversational memory
* A polished Streamlit UI
* Cloud deployment via HuggingFace Spaces
* Modular code design suitable for research & production
* Fully open-source, lightweight, extensible

---

# **Live Demo**

👉 **[https://chengwu1210-multidocrag.hf.space/](https://chengwu1210-multidocrag.hf.space/)**
- No setup needed — upload PDFs and start asking questions.

---

# **UI Preview**

<div align="center">
  <img src="pic/1.png" width="90%">
  <br><br>
  <img src="pic/2.png" width="90%">
  <br><br>
  <img src="pic/3.png" width="90%">
</div>

---

# **System Overview**

The MultiDocRAG pipeline:

1. Upload PDFs
2. Extract + chunk text
3. Generate embeddings
4. Build a FAISS vector index
5. Retrieve top-k relevant chunks
6. Construct a grounded RAG prompt
7. LLM (via API) generates the final answer
8. Conversation memory improves multi-turn reasoning

---

# **Architecture**

```
                  ┌────────────────────┐
                  │     PDF Upload     │
                  └─────────┬──────────┘
                            │
                   Text Extraction
                            │
                            ▼
                   Chunking + Embeddings
                            │
                            ▼
                ┌──────────────────────┐
                │   FAISS Vector Index │
                └───────┬──────────────┘
                        │ Retrieval (k)
                        ▼
             Retrieved Context Chunks
                        │
                        ▼
               RAG Prompt Construction
                        │
                        ▼
             ┌────────────────────────┐
             │ External API LLM Model │
             └────────────────────────┘
                        │
                      Output
```

---

# **Conversational Memory**

MultiDocRAG implements a **session-level sliding-window memory** mechanism:

* Stores the most recent user/assistant turns
* Injects this history into each new prompt
* Enables follow-up reasoning
* Helps the model maintain dialogue continuity

Memory is intentionally lightweight (not training-dependent) to ensure:

* Predictable behavior
* Fast inference
* Good alignment with retrieval context

Screenshot example:

<div align="center">
  <img src="pic/3.png" width="90%">
</div>

---

# **Streamlit UI Features**

### ✔ Upload PDFs (multi-upload supported)

### ✔ Rebuild or reuse FAISS index

### ✔ Adjust LLM sampling parameters (temperature, top-p)

### ✔ Choose Baseline mode or RAG mode

### ✔ Visualize top retrieved chunks

### ✔ Inspect full prompts for debugging

### ✔ Track conversation history

---

# **Repository Structure**

```
MultiDocRAG/
│
├── src/
│   ├── retriever.py           # FAISS retrieval system
│   ├── llm_api.py             # External LLM API wrapper (Groq/OpenAI/etc.)
│   └── __init__.py
│
├── streamlit_app.py           # The main UI application
├── rag_pipeline.py            # Command-line RAG pipeline
├── requirements.txt           
│
├── index_store/               # Auto-generated vector index
│   ├── embeddings.npy
│   ├── chunks.json
│   ├── faiss.index
│   └── meta.json
│
├── pic/                       # Screenshots for README
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
│
└── README.md
```

---

# **How to Run Locally**

### 1. Clone the repo

```bash
git clone https://github.com/ChengWu-Data/MultiDocRAG.git
cd MultiDocRAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run streamlit_app.py
```

---

# **Command-Line RAG Pipeline**

```
python rag_pipeline.py --question "What is the paper about?"
```

This:

* Loads the FAISS index
* Retrieves relevant chunks
* Builds a RAG prompt
* Calls the LLM API
* Outputs the final grounded answer

---

# **LLM Model Options**

Your system works with:

* Groq (fast, free-tier available)
* OpenAI API
* Any Open LLM with a compatible chat completion endpoint

Model selection happens in:

```
src/llm_api.py
```

---

# **Customization & Extensions**

You can easily plug in:

* ❇️ Better embedding models (E5-large, BGE-large)
* ❇️ Rerankers (Cross-encoders, ColBERT)
* ❇️ Citation generation
* ❇️ Richer multi-step memory
* ❇️ Larger LLM backends

The architecture is intentionally modular for experimentation.

---

# **For Coursework Submission (AML Final Project)**

This project satisfies:

* ✔ RAG pipeline
* ✔ LLM integration
* ✔ Memory mechanism
* ✔ Retrieval component
* ✔ Explanation of architecture
* ✔ Full working demo
* ✔ Cloud deployment
* ✔ UI for user interaction

Everything required is implemented, working, and well-documented.

---

# 📄 License

MIT License.
