# Assignment 3 – Autonomous Multi-Doc Financial Analyst

State-aware RAG system built with **LangGraph** (Tasks B–E) and a **ReAct** fallback agent (Task A).

---

## Prerequisites

- Python 3.11
- A `.env` file (copy from `.env_example` and fill in your API key)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env_example .env            # then edit .env with your API key
```

---

## Directory Layout

```
.
├── data/                    # Source PDFs (Apple + Tesla)
├── chroma_db/               # Auto-generated vector databases (gitignored)
│   ├── minilm/              # Indexes built with MiniLM embeddings
│   │   ├── apple/
│   │   └── tesla/
│   └── mpnet/               # Indexes built with MPNet embeddings
│       ├── apple/
│       └── tesla/
├── config.py                # LLM factory + embedding config
├── build_rag.py             # PDF → ChromaDB ETL pipeline
├── langgraph_agent.py       # LangGraph agent (Tasks A–E)
├── evaluator.py             # LLM-as-Judge benchmark
├── requirements.txt
└── .env_example
```

---

## Step 1: Build the Vector Databases

```bash
# Default (MiniLM embeddings, chunk_size=1000)
python build_rag.py

# Alternate embedding model for benchmarking
EMBEDDING_MODEL=mpnet python build_rag.py

# Larger chunk size experiment
CHUNK_SIZE=2000 python build_rag.py --force
```

---

## Step 2: Run the Evaluation

```bash
# LangGraph agent (default)
python evaluator.py

# ReAct / LangChain agent
python evaluator.py --mode LEGACY

# Side-by-side embedding model comparison
python evaluator.py --compare-models
```

Results are printed to the console **and** saved to `evaluation_log_YYYYMMDD_HHMM.txt`.

---

## Tasks Implemented

| Task | Component | Description |
|------|-----------|-------------|
| A | `run_legacy_agent` | ReAct prompt with English-only, year-precision, and honesty constraints |
| B | `retrieve_node` | LLM router: classifies query into `apple / tesla / both / none` |
| C | `grade_documents_node` | Binary relevance grader: `yes` → generate, `no` → rewrite |
| D | `rewrite_node` | Rewrites vague queries into precise financial terminology |
| E | `generate_node` | Generates cited answers; says "I don't know" when info is missing |

---

## Benchmark Experiments (for Report)

### 1. Embedding Model Comparison
| Model | Key | Characteristics |
|-------|-----|----------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | `minilm` | Smaller, multilingual, fast |
| `all-mpnet-base-v2` | `mpnet` | Larger, English-focused, higher quality |

Run with `python evaluator.py --compare-models`.

### 2. Chunk Size Experiment
| Config | `chunk_size` | Trade-off |
|--------|-------------|-----------|
| Small  | 500–1000    | Higher precision, may miss full tables |
| Default | 1000       | Balanced |
| Large  | 2000        | Better context completeness for large tables (e.g., Balance Sheet), lower precision |

Rebuild with `CHUNK_SIZE=2000 python build_rag.py --force`, then re-run evaluator.

### 3. LangGraph vs. LangChain (ReAct)
Run the same test suite with `--mode GRAPH` and `--mode LEGACY` and compare scores.
