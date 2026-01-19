# 🛍️ Wholesale Product Retrieval System

LLM-enhanced hybrid search + reranking for marketplace product retrieval, evaluated on Amazon's ESCI benchmark with wholesale-augmented queries.

## 🎯 Project Overview

This project builds a production-style product search system that combines:
- **LLM Query Understanding** (Flan-T5) — extracts intent, product type, attributes
- **Hybrid Retrieval** (BM25 + Dense Embeddings) — best of lexical + semantic search
- **Learning-to-Rank Reranking** (XGBoost) — optimizes final ranking

Built for wholesale/B2B marketplace scenarios (like [Faire](https://www.faire.com/)).

## 📊 Results

| Configuration | NDCG@10 | MRR@10 | Recall@50 | Latency (p95) |
|---------------|---------|--------|-----------|---------------|
| BM25 only (baseline) | — | — | — | — |
| Dense only | — | — | — | — |
| Hybrid (BM25 + Dense) | — | — | — | — |
| + LLM query expansion | — | — | — | — |
| **Full pipeline** | — | — | — | — |

*Results will be filled after running evaluation.*

## 🏗️ Architecture

```
User Query → Query Understanding → Hybrid Retrieval → Reranker → Results
                  (LLM)             (BM25 + FAISS)     (XGBoost)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter project
git clone <your-repo-url>
cd wholesale-product-retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

```bash
cd notebooks
jupyter notebook
```

Run notebooks sequentially:
1. `01_data_prep.ipynb` — Load ESCI, create augmented queries
2. `02_query_understanding.ipynb` — Build LLM module
3. `03_indexing.ipynb` — Create BM25 + FAISS indices
4. `04_retrieval.ipynb` — Implement hybrid retrieval
5. `05_reranking.ipynb` — Train XGBoost ranker
6. `06_evaluation.ipynb` — Run ablation studies

### 3. Launch Demo

```bash
streamlit run app/streamlit_demo.py
```

## 📁 Project Structure

```
wholesale-product-retrieval/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml           # All hyperparameters
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_query_understanding.ipynb
│   ├── 03_indexing.ipynb
│   ├── 04_retrieval.ipynb
│   ├── 05_reranking.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── augmentation.py       # Wholesale query augmentation
│   ├── query_understanding.py
│   ├── retrieval.py
│   ├── reranker.py
│   └── evaluation.py
├── app/
│   └── streamlit_demo.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── ablation_results.csv
    └── figures/
```

## 📈 Dataset

**Amazon ESCI** (Shopping Queries Dataset)
- Source: [Hugging Face](https://huggingface.co/datasets/tasksource/esci)
- ~1.8M products, ~130K labeled queries
- Labels: Exact / Substitute / Complement / Irrelevant

**Wholesale Augmentation**
- 25% of queries augmented with B2B signals
- Templates: "bulk", "wholesale", "case pack", quantity signals

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Query Understanding | Flan-T5-base |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Index | FAISS |
| Lexical Search | rank-bm25 |
| Reranker | XGBoost (LambdaMART) |
| Demo | Streamlit |

## 📝 License

MIT

## 🙏 Acknowledgments

- Amazon ESCI dataset (KDD Cup 2022)
- Faire's tech blog for marketplace search insights