# TruthGate: FastAPI Documentation RAG System

## Overview

TruthGate is a Retrieval-Augmented Generation (RAG) system built on top of the FastAPI documentation. The system automatically crawls documentation pages, chunks and embeds content, stores embeddings in a vector database, retrieves relevant context for user queries, reranks results using a cross-encoder, and generates grounded responses using a local Large Language Model (LLM).

The project is designed to answer questions using only the FastAPI documentation while supporting refusal behavior for unanswerable questions and detecting false-premise queries.

---

# Goal

The objective of this project is to build a documentation-grounded question answering system that:

* Retrieves information exclusively from FastAPI documentation.
* Generates answers using retrieved documentation context.
* Refuses to answer when relevant information is unavailable.
* Identifies false-premise questions that contradict documentation.
* Operates using local embedding, retrieval, reranking, and LLM components.

---

# System Architecture

```text
FastAPI Documentation
          │
          ▼
     Web Scraper
          │
          ▼
     JSON Documents
          │
          ▼
  Text Chunking
(Recursive Splitter)
          │
          ▼
 Embedding Generation
 (BGE Small v1.5)
          │
          ▼
      ChromaDB
          │
          ▼
  Similarity Retrieval
          │
          ▼
 Cross-Encoder Reranking
          │
          ▼
 Answerability Classification
          │
          ▼
      Qwen3:4B
          │
          ▼
 Generated Response
```

---

# Implementation

## 1. Documentation Crawling

The crawler performs breadth-first traversal of FastAPI documentation pages and extracts textual content from HTML pages.

### Features

* Domain-restricted crawling
* URL normalization
* HTML parsing using BeautifulSoup
* Structured JSON output generation
* Duplicate URL filtering

### Output

```json
{
  "title": "Path Parameters",
  "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
  "content": "..."
}
```

Generated file:

```text
data/fastapi_docs.json
```

---

## 2. Chunking

Documentation content is segmented using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

### Motivation

* Preserves context across chunk boundaries.
* Improves retrieval quality.
* Keeps chunks within embedding model limits.

---

## 3. Embedding Generation

Embedding model:

```text
BAAI/bge-small-en-v1.5
```

Characteristics:

* Dense semantic embeddings
* 384-dimensional vectors
* Efficient inference
* Strong retrieval performance

---

## 4. Vector Storage

Vector database:

```text
ChromaDB
```

Stored information:

* Chunk text
* Metadata
* Embeddings

Metadata includes:

```json
{
  "url": "...",
  "section": "...",
  "chunk_index": 0
}
```

---

## 5. Retrieval Pipeline

For each query:

### Step 1

Retrieve top-k candidate chunks from ChromaDB.

### Step 2

Rerank candidates using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves retrieval precision by scoring query-document relevance.

### Step 3

Build contextual prompt using top-ranked chunks.

---

## 6. Classification Layer

Before generating answers, the system classifies queries into:

```text
ANSWERABLE
UNANSWERABLE
FALSE_PREMISE
```

This step prevents hallucinated responses and supports grounded answering.

---

## 7. Answer Generation

Local LLM:

```text
Qwen3:4B
```

Executed through:

```text
Ollama
```

The model receives:

* User question
* Retrieved context
* Citation information

and generates a grounded response.

---

# Project Structure

```text
TruthGate/
│
├── scrape.py
├── build_index.py
├── query_engine.py
├── run_eval.py
├── cli.py
│
├── data/
│   └── fastapi_docs.json
│
├── db/
│   └── chroma/
│
└── eval/
    └── questions.json
```

---

# Running the Project

## Step 1: Crawl Documentation

```bash
python scrape.py
```

Output:

```text
data/fastapi_docs.json
```

---

## Step 2: Build Vector Index

```bash
python build_index.py
```

Output:

```text
db/chroma/
```

---

## Step 3: Run Interactive CLI

```bash
python cli.py
```

Example:

```text
Question> How do query parameters work in FastAPI?
```

---

## Step 4: Run Evaluation

```bash
python run_eval.py
```

---

# Evaluation

Evaluation is performed using a manually curated dataset of documentation-related questions.

Question categories include:

* Answerable
* Unanswerable
* False Premise
* Adversarial

Metrics measured:

* Overall Accuracy
* Refusal Precision
* Refusal Recall
* False Premise Accuracy
* Average Query Latency

### Current Results

```text
Total Questions: 60

Accuracy: 16/48 = 0.33
Refusal Precision: 0.48
Refusal Recall: 1.00
False Premise Accuracy: 0.00
Average Latency: 0.34 seconds
```

---

# Advantages 
* Fully local pipeline
* No external API costs
* Grounded answers using retrieved documentation
* Citation support
* Refusal mechanism

---

# Limitations

* Retrieval quality depends heavily on scraper output quality.
* Navigation and boilerplate content may affect retrieval accuracy.
* Current refusal threshold is conservative and may reject answerable questions.
* False-premise classification requires further tuning.

---




