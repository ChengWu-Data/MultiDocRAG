# 🧠 **MultiDocRAG**

### *A Retrieval-Augmented Multi-Document Reasoning Assistant*

MultiDocRAG is an LLM-powered system designed for **cross-document reasoning**.
It enables users to upload multiple PDFs—papers, reports, articles—and ask grounded questions that require **comparison, synthesis, and multi-document understanding**.

The system integrates:

* **Multi-document ingestion & chunking**
* **Vector-based retrieval with embeddings**
* **LLM reasoning over retrieved evidence**
* **Optional conversational memory** for context continuity
* **A full evaluation pipeline** comparing baseline LLM vs RAG-enhanced performance

This project was developed as part of **COMS 4995 – Applied Machine Learning** at Columbia.

---

## 🚀 **Key Features**

### **📄 Multi-Document Ingestion**

Upload several PDFs at once.
The system automatically extracts text, segments it into semantic chunks, and stores them in a vector database.

### **🔍 Retrieval-Augmented Generation (RAG)**

Queries are grounded in the uploaded documents through top-k similarity search.
Responses include **citations** to the most relevant chunks.

### **🧩 Cross-Document Reasoning**

Designed to answer questions like:

* *“Compare method A in Paper 1 and method B in Paper 2.”*
* *“Summarize common limitations across these documents.”*
* *“What does Paper 3 say about X, and how does it differ from Paper 1?”*

### **🧠 Optional Memory Module**

Keeps track of previous interactions and user preferences to improve coherence in multi-turn conversations.

### **📊 Evaluation Framework**

We rigorously compare:

* **Baseline LLM** (no RAG, single-pass prompting)
* **RAG-based system**
* **RAG + Memory system**

Using metrics such as:

* Relevance
* Faithfulness
* Ability to cite correct documents
* Multi-document synthesis quality

### **💻 Clean Demo Interface**

A simple UI / notebook demo allows:

1. PDF upload
2. Query input
3. Retrieval visualization
4. Final synthesized answer with citations

---

## 🏗️ **System Architecture**

```
                ┌────────────────────────┐
                │        PDFs (n)        │
                └────────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │  Document Ingestion   │
                 │ (extraction + chunks) │
                 └────────────┬──────────┘
                             ▼
                 ┌───────────────────────┐
                 │     Embeddings        │
                 │   (vector database)   │
                 └────────────┬──────────┘
                             ▼
                 ┌───────────────────────┐
                 │      Retrieval        │
                 └────────────┬──────────┘
                             ▼
                 ┌───────────────────────┐
                 │  LLM Reasoning Layer  │
                 │ (RAG + Memory + CoT)  │
                 └────────────┬──────────┘
                             ▼
                 ┌───────────────────────┐
                 │     Final Answer      │
                 │     + Citations       │
                 └───────────────────────┘
```

---

## 📦 **Repository Structure**

```
MultiDocRAG/
│
├── src/
│   ├── ingestion/         # PDF loading, extraction, chunking
│   ├── embeddings/        # Embedding model wrappers
│   ├── retrieval/         # Vector search & reranker
│   ├── llm/               # Prompting, reasoning, memory, CoT
│   ├── evaluation/        # Baseline vs RAG comparisons
│   ├── demo/              # Notebook / Streamlit app
│   └── utils/             # Helper functions
│
├── data/                  # Sample PDFs (if allowed)
│
├── experiments/           # Results, tables, qualitative examples
│
├── README.md
├── requirements.txt
├── LICENSE (MIT)
└── .gitignore
```

---

## 🧪 **Evaluation Overview**

We evaluate on tasks including:

* **Cross-document QA**
* **Comparative analysis**
* **Evidence attribution**
* **Long-context question consistency**

Example evaluation question:

> *“How does the methodology in Paper A differ from Paper B in terms of data assumptions and model constraints?”*

The system generates:

* Answer with synthesized explanation
* Citations for each referenced document
* Evidence snippets retrieved

---

## ▶️ **Demo Instructions**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Run the demo app**

```
streamlit run app.py
```

### **3. Upload PDFs and start asking questions**

---

## 🤝 **Steps**

* **Document ingestion & retrieval**
* **LLM logic (RAG + reasoning + memory)**
* **System integration & demo**
* **Evaluation + report (baseline vs RAG/memory, experiments, tables, write-up)**

---

## 📜 **License**

MIT License

---

# 🎉 **MultiDocRAG: Turning Multiple PDFs Into One Coherent Answer**


