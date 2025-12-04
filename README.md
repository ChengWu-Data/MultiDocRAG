# MultiDocRAG

**MultiDocRAG** is a retrieval-augmented question answering pipeline designed to support reasoning across multiple documents.
It includes document ingestion, vector retrieval, a safe LLM reasoning layer, and an evaluation setup.
The implementation is fully reproducible through notebooks and an executable script.

This project was developed as part of *COMS 4995 — Applied Machine Learning* at Columbia University.

---

## System Overview

The project is organized into four main components, aligned with the intended team structure:

1. **Document Ingestion & Retrieval**
   PDF extraction, chunking, embedding generation, and vector index construction.

2. **LLM Logic**
   Prompt design, retrieval-augmented generation (RAG), and a safe generation wrapper compatible with larger models.

3. **System Integration & Demo**
   Execution notebooks and a command-line pipeline that connect all components.

4. **Evaluation & Report**
   Baseline vs. RAG comparisons, qualitative outputs, and the foundation for the final project report.

---

## 1. Document Ingestion & Retrieval

Notebook `01_ingestion_retrieval.ipynb` performs:

* PDF text extraction
* Chunking with configurable size and overlap
* Embedding generation using `all-MiniLM-L6-v2`
* Building and saving a FAISS index

Artifacts are stored in:

```
index_store/
├── chunks.json
├── embeddings.npy
├── faiss.index
└── meta.json
```

The retriever logic is implemented in:

```
src/retriever.py
```

---

## 2. LLM Logic (Baseline and RAG)

Notebook `02_llm_rag.ipynb` contains:

* Baseline LLM answering (no retrieved context)
* Retrieval-augmented answering using retrieved chunks
* Construction of RAG prompts
* A robust `generate_from_model()` implementation that:

  * automatically respects model context limits
  * trims long prompts safely
  * supports temperature and nucleus sampling
  * works with small and large open-source models
  * runs fully offline without API keys

This module is compatible with future model upgrades (e.g., LLaMA, Mistral, Gemma, TinyLlama, Phi-3).

---

## 3. System Integration & Demo

The project includes two ways to run the pipeline:

### Notebook Workflow

* `01_ingestion_retrieval.ipynb` builds the index.
* `02_llm_rag.ipynb` performs baseline and RAG comparisons.

### Command-Line Script

The file `rag_pipeline.py` provides a simple interface for running RAG directly:

```
python rag_pipeline.py --question "Your question here"
```

This script:

1. Loads the vector index
2. Retrieves relevant evidence
3. Builds a contextual prompt
4. Produces an answer through the LLM

This is useful for demos, reproducibility, and automated evaluation.

---

## 4. Evaluation

Notebook 02 provides:

* Side-by-side comparison of Baseline vs. RAG LLM responses
* Multi-question evaluation loop
* Qualitative examination of grounding and relevance

The structure supports extension to:

* attribution-based evaluation
* similarity-based scoring
* ablation studies (e.g., varying k, model choice, sampling settings)

These elements form the foundation for the final project report and presentation.

---

## Repository Structure

```
MultiDocRAG/
│
├── rag_pipeline.py                  # Command-line RAG pipeline
├── requirements.txt                 # Python dependencies
│
├── index_store/                     # Vector database
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
│   └── retriever.py
│
├── LICENSE
├── .gitignore
└── README.md
```

---

## Installation

Install dependencies using:

```
pip install -r requirements.txt
```

---

## How to Run

### Build the Index

Open:

```
notebooks/01_ingestion_retrieval.ipynb
```

and follow the steps to:

* upload PDFs
* extract and chunk text
* build and save the FAISS index

### Run RAG Answering

Open:

```
notebooks/02_llm_rag.ipynb
```

to compare Baseline vs. RAG responses interactively.

### Command-Line Usage

```
python rag_pipeline.py --question "What are the main sources of interest rate risk?"
```

---

## Future Extensions

The project can be extended with:

* larger or domain-specific embedding models
* long-context or fine-tuned LLMs
* multi-turn conversational memory
* citation attribution
* a Streamlit or web-based interface
* quantitative evaluation metrics

---

## License

MIT License.

