# **MultiDocRAG — Multi-Document Retrieval-Augmented Generation System**

**MultiDocRAG** is a retrieval-augmented question answering system capable of ingesting multiple documents, building a vector database, and performing context-grounded LLM reasoning.
The system includes:

* A full ingestion → retrieval → RAG pipeline
* A clean Streamlit demo UI
* Notebook workflows for reproducibility
* A lightweight modular architecture for future expansion

This project was developed for **COMS 4995 — Applied Machine Learning (Fall 2025)** at Columbia University.

---

# **Live System Overview**

MultiDocRAG enables users to:

1. **Ingest arbitrary PDFs**
2. **Embed and index text chunks using FAISS**
3. **Retrieve relevant context for a user query**
4. **Generate answers using a local or open-source LLM**
5. **Compare Baseline vs RAG output in real time**
6. **Use a Streamlit UI to explore retrieval results & model answers**

The system is fully offline and works with open-source models (e.g., Qwen2.5, Mistral, LLaMA-based checkpoints).

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
             │  Local LLM Generation  │
             └────────────────────────┘
                        │
                      Output
```

Modules are cleanly separated so each step can be replaced or improved independently.

---

# **Features**

### **Multi-document ingestion & retrieval**

* PDF parsing
* Chunking with configurable window size & overlap
* Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* FAISS vector index with metadata fields

### **Baseline vs RAG LLM reasoning**

* Baseline model answering (no context)
* RAG enhanced answering (retrieved context injected)
* Automatic prompt trimming for long inputs
* Temperature and top-p sampling controls

### **Streamlit Demo Interface (New)**

`app.py` provides a polished UI:

* Upload PDFs
* Build or reload existing FAISS index
* Ask questions via text box
* View retrieved text chunks
* Compare Baseline vs RAG answers
* Real-time error handling (missing index, empty PDFs, etc.)

### **Modular & Extensible**

* Swap embedding model
* Swap LLM backbone
* Add reranking
* Add citations
* Add conversational memory

---

# **Repository Structure**

```
MultiDocRAG/
│
├── app.py                          # Streamlit user interface (NEW)
├── rag_pipeline.py                 # Command-line RAG pipeline
├── requirements.txt                 
│
├── index_store/                    # Generated vector index (created at runtime)
│   ├── chunks.json
│   ├── embeddings.npy
│   ├── faiss.index
│   └── meta.json
│
├── notebooks/
│   ├── 01_ingestion_retrieval.ipynb
│   └── 02_llm_rag.ipynb
│
├── src/
│   ├── __init__.py
│   └── retriever.py                # Retriever class with FAISS logic
│
├── LICENSE
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ChengWu-Data/MultiDocRAG.git
cd MultiDocRAG
pip install -r requirements.txt
```

---

# **How to Run the System**

## **Option 1: Run the Streamlit Demo (Recommended)**

```bash
streamlit run app.py
```

This launches an interactive UI where you can:

1. Upload PDFs
2. Build the FAISS index
3. Enter questions
4. Compare baseline vs RAG responses
5. View retrieved context

---

## **Option 2: Reproduce with Notebooks**

### **Build the index**

Open:

```
notebooks/01_ingestion_retrieval.ipynb
```

and follow instructions to:

* upload documents
* extract and chunk text
* generate embeddings
* save FAISS index to `index_store/`

### **Test Baseline vs RAG answering**

Open:

```
notebooks/02_llm_rag.ipynb
```

---

## **Option 3: Command-Line Usage**

```bash
python rag_pipeline.py --question "What is the main idea of the paper?"
```

This:

1. Loads FAISS index
2. Retrieves k chunks
3. Builds a RAG prompt
4. Produces a grounded answer

---

# **Evaluation**

The system supports:

* Baseline vs RAG qualitative comparison
* Retrieval quality inspection
* Multi-question loops
* Chunk relevance visualization

Planned extensions:

* automatic attribution
* Rouge/BLEU evaluation
* embedding ablations (`k`, model choice)
* reranking

---

# **Future Extensions**

* Introduce document-level memory
* Add citation markers in generated answers
* Support images or tables (multi-modal RAG)
* Provide a REST API / Docker deployment
* Integrate larger embeddings (e.g., E5-large)
* Use LLM-as-a-judge for automated evaluation

---

# License

MIT License.

