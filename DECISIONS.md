# DECISIONS

## Data Collection

* Stored scraped pages in a local JSON file.
* Kept page title, URL, and cleaned content for each document.
* Re-scraping the documentation allows the knowledge base to be refreshed easily.

## Chunking Strategy

* Used `RecursiveCharacterTextSplitter`.
* Chunk size: 800 characters.
* Chunk overlap: 150 characters.
* Smaller chunks improve retrieval precision while overlap helps preserve context across chunk boundaries.

## Embedding Model

* Selected `BAAI/bge-small-en-v1.5` for document and query embeddings.
* Chosen because it provides good retrieval performance while remaining lightweight enough for local execution.
* Ensured the same model is used during indexing and querying to avoid embedding mismatch issues.

## Vector Database

* Used ChromaDB for storing embeddings and metadata.
* Chosen because it is lightweight, open source, and easy to run locally.
* Stores document chunks, embeddings, URLs, and section information.

## Retrieval Pipeline

* Convert user query into an embedding.
* Perform similarity search in ChromaDB.
* Retrieve top-k candidate chunks.
* Return the most relevant chunks for reranking.

## Reranking

* Added a Cross Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) after retrieval.
* Improves ranking quality by evaluating query-document pairs directly.
* Helps surface the most relevant chunks before answer generation.

## Refusal Mechanism

* Introduced support for:

  * ANSWER
  * REFUSAL
  * FALSE_PREMISE
* Prevents the system from generating unsupported answers.
* Encourages evidence-based responses instead of hallucinations.

## LLM Selection

* Used Ollama with `qwen3:4b`.
* Chosen because it runs locally and does not require paid APIs.
* Used for:

  * Answerability classification
  * False-premise detection
  * Final answer generation

## Evaluation

* Created a manually written evaluation set.
* Included:

  * Answerable questions
  * Unanswerable questions
  * False-premise questions
  * Adversarial questions
* Measured:

  * Accuracy
  * Refusal Precision
  * Refusal Recall
  * False Premise Accuracy
  * Latency

## Challenges Faced

* Initial retrieval quality was poor due to embedding inconsistencies.
* Query embeddings and indexed embeddings were not generated using the same configuration.
* Ollama output often contained reasoning text instead of direct labels.
* Retrieval thresholds required tuning to avoid excessive refusals.
* Scraping quality directly affected retrieval performance.

## Improvements

* Cache embedding and reranking models globally to reduce latency.
* Improve document cleaning during scraping.
* Tune refusal thresholds using evaluation results.
* Expand the evaluation dataset.
* Improve answer formatting and citation quality.
