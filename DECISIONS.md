# Notes and Challenges

## Overview of the System

The project follows a Retrieval-Augmented Generation (RAG) style pipeline built entirely using local components.

1. **scrape.py**

   * Crawls the FastAPI documentation website.
   * Extracts page content and stores it in a structured JSON file.

2. **build_index.py**

   * Splits documentation pages into chunks.
   * Generates embeddings using BAAI/bge-small-en-v1.5.
   * Stores embeddings and metadata in ChromaDB.

3. **query_engine.py**

   * Converts user queries into embeddings.
   * Retrieves relevant chunks from ChromaDB.
   * Reranks results using a Cross Encoder.
   * Uses Ollama (Qwen3) for classification and answer generation.

4. **cli.py**

   * Provides a simple command-line interface for asking questions.

5. **run_eval.py**

   * Runs the evaluation dataset and reports system metrics.

---

## Initial Evaluation Results

During the first dry run, the system performed very poorly.

* Total Questions: 48
* Accuracy: 16/48 = 0.33
* Refusal Precision: 0.48
* Refusal Recall: 1.00
* False Premise Accuracy: 0.00
* Average Latency: 0.34 seconds

These results indicated that although the refusal mechanism was triggering correctly, the system was failing to answer many valid questions and was unable to detect false-premise queries properly.

---

## Issues 

While debugging the system, several major issues were found:

### Embedding Mismatch

One of the most significant problems was that document embeddings and query embeddings were not being generated consistently.

As a result, retrieval quality was extremely poor even when the answer existed in the documentation.

After correcting the embedding pipeline and rebuilding the index, retrieval quality improved significantly.

### Scraping Quality

The initial scraper collected a large amount of noisy content from navigation menus, translated pages, release notes, and other sections that were not useful for question answering.

This caused retrieval results to become noisy and reduced ranking quality.

Several improvements were made to the scraper, but the dataset still contains noise that affects retrieval performance.

### Ollama Classification Problems

The classifier was instructed to output only:

* ANSWERABLE
* UNANSWERABLE
* FALSE_PREMISE

However, Qwen frequently produced long reasoning traces before the final label.

For example, instead of returning:

ANSWERABLE

it would generate an entire explanation and then end with the label.

Additional parsing logic had to be added to extract the final classification reliably.

### Threshold Tuning

The refusal threshold turned out to be difficult to tune.

A strict threshold caused many valid questions to be rejected.

A loose threshold increased the risk of answering unsupported questions.

Finding a balance remains an ongoing challenge.

### Latency

The system currently reloads multiple models during execution:

* Embedding model
* Cross Encoder
* Ollama inference

This significantly increases response time.

Even after retrieval improvements, latency remains one of the biggest unresolved issues.

---

## A Failing Case

One of the worst failures occurred with the question:

> "How do query parameters work in FastAPI?"

This is a straightforward question directly covered in the FastAPI documentation.

The retriever correctly returned highly relevant chunks from:

* Query Parameters
* Query Parameters and String Validations

However, the classification stage incorrectly marked the query as FALSE_PREMISE or REFUSAL during several runs.

The failure was not caused by retrieval.

The relevant information was successfully retrieved.

The issue occurred in the LLM classification layer, which sometimes ignored instructions and produced inconsistent outputs.

This highlighted that the retrieval pipeline was functioning correctly while the decision layer remained unreliable.

---

## Current Status

The retrieval pipeline is now substantially better than during the first evaluation run.

Relevant chunks are being retrieved consistently, reranking is working correctly, and the embedding mismatch issue has been resolved.

However, a few important problems still remain:

* Classification reliability
* Refusal threshold tuning
* Scraping noise
* High latency

I am actively working on these issues, but due to the limited project timeline I was unable to fully resolve them before submission.

The system demonstrates the complete architecture and workflow successfully, but there is still room for improvement in both accuracy and efficiency.
