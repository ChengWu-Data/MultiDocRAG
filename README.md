# **MultiDocRAG**

### *A Retrieval-Augmented Multi-Document Question Answering System*

MultiDocRAG is a lightweight Retrieval-Augmented Generation (RAG) pipeline for **multi-document question answering**.
It supports PDF ingestion, semantic chunking, FAISS-based retrieval, and safe LLM reasoning—all implemented through clean and reproducible notebooks.

This project was developed as part of **COMS 4995 — Applied Machine Learning** at Columbia University.

---

# Features

### **1. PDF Ingestion & Chunking**

Notebook **`01_ingestion_retrieval.ipynb`** extracts text from multiple PDFs, splits them into overlapping chunks, and saves them into a searchable index.

### **2. Vector Retrieval with FAISS**

A custom retriever (`src/retriever.py`) performs embedding computation and FAISS indexing.
The resulting vector database lives in:

```
index_store/
├── chunks.json
├── embeddings.npy
├── faiss.index
└── meta.json
```

### **3. Retrieval-Augmented Generation (RAG)**

Notebook **`02_llm_rag.ipynb`** loads:

* the retriever
* an open-source causal LLM
* a safe, context-aware generation wrapper
* a unified QA interface (`answer_question`)

Users can compare:

* **Baseline LLM responses** (no context)
* **RAG-enhanced responses** (grounded in retrieved evidence)

### **4. Safe, Scalable LLM Usage**

The custom `generate_from_model()`:

* automatically avoids context-length overflow
* supports nucleus sampling (`top_p`) and temperature
* works with small or large models
* runs entirely offline (no API keys required)

---

# Repository Structure

```
MultiDocRAG/
│
├── index_store/                     # Vector DB built from PDFs
│   ├── chunks.json
│   ├── embeddings.npy
│   ├── faiss.index
│   └── meta.json
│
├── notebooks/
│   ├── 01_ingestion_retrieval.ipynb # Build index from documents
│   └── 02_llm_rag.ipynb             # RAG pipeline + QA comparison
│
├── src/
│   ├── __init__.py
│   └── retriever.py                 # MultiDocRetriever implementation
│
├── LICENSE
├── .gitignore
└── README.md
```

---

# How to Use

### **1. Install Dependencies**

```
pip install transformers sentence-transformers faiss-cpu pypdf
```

### **2. Build Retrieval Index (Notebook 01)**

* Upload PDFs
* Extract + chunk the text
* Generate embeddings
* Save index to `index_store/`

### **3. Run RAG QA (Notebook 02)**

* Load the retriever
* Load an open-source LLM
* Ask multi-document questions
* Compare baseline vs RAG answers

---

# Example Query

> **“What are the main sources of interest rate risk discussed across these papers?”**

The system:

1. Retrieves top-k relevant chunks
2. Builds a contextual RAG prompt
3. Generates an answer grounded in retrieved evidence

---

# Future Extensions

The repo is structured to expand to:

* Larger LLMs (LLaMA 3, Mistral, Gemma, TinyLlama, Phi-3, etc.)
* Stronger embedding models (BGE, E5, GTE)
* Multi-turn conversational memory
* Citation-aware answering
* Streamlit UI for real-time PDF QA

---

# License

MIT License.

---

