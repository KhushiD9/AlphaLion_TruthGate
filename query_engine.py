import json
import os
import shutil
import subprocess
import time
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
import torch

load_dotenv()

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "db/chroma"))
COLLECTION_NAME = "fastapi_fastapi_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_PATH = os.environ.get("OLLAMA_PATH", "ollama")
REFUSAL_THRESHOLD = float(os.environ.get("REFUSAL_THRESHOLD", 0.35))
TOP_K = int(os.environ.get("TOP_K", 10))


def run_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 180) -> str:
    if shutil.which(OLLAMA_PATH) is None:
        raise RuntimeError(
            f"Ollama executable not found at '{OLLAMA_PATH}'. "
            "Install Ollama or set OLLAMA_PATH to the executable path."
        )
    
    import tempfile
    # Write prompt to temp file and pipe it to ollama
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(prompt)
        temp_file = f.name
    
    try:
        with open(temp_file, 'r', encoding='utf-8') as f:
            result = subprocess.run(
                [OLLAMA_PATH, "run", model],
                stdin=f,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Ollama executable not found at '{OLLAMA_PATH}'. "
            "Install Ollama or set OLLAMA_PATH to the executable path."
        ) from exc
    finally:
        import os as os_module
        if os_module.path.exists(temp_file):
            os_module.remove(temp_file)
    
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()


def normalize_answer(text: str) -> str:
    return text.strip().replace("\n", " ")


def classification_prompt(question: str, context: str) -> str:
    return f"""
Answer with ONLY ONE WORD.

ANSWERABLE
UNANSWERABLE
FALSE_PREMISE

Question:
{question}

Context:
{context}
"""


def render_answer_prompt(question, citations, context):
    citation_text = "\n".join(
        [f"[{i+1}] {item['section']} - {item['url']}"
         for i, item in enumerate(citations)]
    )

    return (
        "Answer using ONLY the provided context.\n"
        "Do not think aloud.\n"
        "Do not explain your reasoning.\n"
        "Do not write Thinking...\n"
        "Give the final answer directly.\n\n"
        f"Question:\n{question}\n\n"
        f"Citations:\n{citation_text}\n\n"
        f"Context:\n{context}\n"
    )


def load_chroma_collection() -> Any:
    client = Client(settings=Settings(is_persistent=True, persist_directory=str(CHROMA_DIR)))
    return client.get_or_create_collection(name=COLLECTION_NAME)


class BgeTextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def embed_text(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        if torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            last_hidden_state = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            divisor = mask.sum(dim=1).clamp(min=1e-9)
            embedding = (summed / divisor).cpu().tolist()[0]
        return embedding


def format_query_results(query_results: dict[str, list]) -> list[dict]:
    docs = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]
    results = []
    for doc, meta, distance in zip(docs, metadatas, distances):
        score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
        results.append({"content": doc, "metadata": meta, "distance": distance, "score": score})
    return results


def rerank_with_cross_encoder(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    reranker = CrossEncoder(CROSS_ENCODER)
    pairs = [[question, candidate["content"]] for candidate in candidates]
    scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def ask_question(question: str) -> dict[str, Any]:
    start_time = time.perf_counter()
    collection = load_chroma_collection()

    embedder = BgeTextEmbedder()
    query_embedding = embedder.embed_text(question)

    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    candidates = format_query_results(query_results)
    if not candidates:
        return {
            "type": "REFUSAL",
            "message": "Not answerable from docs",
            "citations": [],
            "latency": time.perf_counter() - start_time,
        }

    # checking the rectrieved chunks before rerancking
    # print("\nQUESTION:", question)

    for i, candidate in enumerate(candidates[:5]):
        # print(f"\nRESULT {i+1}")
        # print("Distance:", candidate["distance"])
        # print(candidate["metadata"]["url"])
        # print(candidate["content"][:500])
        pass

    top_distance = candidates[0]["distance"]

    # if top_distance is None or top_distance > REFUSAL_THRESHOLD:
    #     return {
    #         "type": "REFUSAL",
    #         "message": "Not answerable from docs",
    #         "citations": [],
    #         "latency": time.perf_counter() - start_time,
    #     }
    # print("Top distance:", top_distance)

    reranked = rerank_with_cross_encoder(question, candidates)
    chosen = reranked[:5]
    context = "\n\n".join([item["content"] for item in chosen])
    top_context = "\n\n".join([item["content"] for item in chosen[:3]])
    classifier_input = classification_prompt(question, top_context)

    classification = normalize_answer(
        run_ollama(classifier_input)
    ).upper()

    # print("\nCLASSIFICATION RAW:")
    # print(classification)

    matches = re.findall(
        r"\b(ANSWERABLE|UNANSWERABLE|FALSE_PREMISE)\b",
        classification,
    )

    predicted_label = matches[-1] if matches else None

    # print("\nPREDICTED LABEL:", predicted_label)

    if predicted_label is None:
        predicted_label = "UNANSWERABLE"

    if predicted_label == "FALSE_PREMISE":
        return {
            "type": "FALSE_PREMISE",
            "message": "The question contains a false premise not supported by FastAPI docs.",
            "citations": [item["metadata"] for item in chosen[:3]],
            "latency": time.perf_counter() - start_time,
        }

    if predicted_label == "UNANSWERABLE":
        return {
            "type": "REFUSAL",
            "message": "Not answerable from docs",
            "citations": [item["metadata"] for item in chosen[:3]],
            "latency": time.perf_counter() - start_time,
        }

    answer_prompt = render_answer_prompt(question, [item["metadata"] for item in chosen[:3]], context)
    answer_text = normalize_answer(run_ollama(answer_prompt))
    return {
        "type": "ANSWER",
        "answer": answer_text,
        "citations": [item["metadata"] for item in chosen[:3]],
        "latency": time.perf_counter() - start_time,
    }

# for the first example getting poor response may be the retrival chunks are bad
# or threshold too strict ..can check others as well
# latency is also an issue here

# there was an issue with the embedding model
# the issue was idexed with BDE but query with diff

if __name__ == "__main__":
    example = "How do query parameters work in FastAPI?"
    response = ask_question(example)
    print(json.dumps(response, indent=2, ensure_ascii=False))
