import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
import torch

CHROMA_DIR = Path("db/chroma")
COLLECTION_NAME = "fastapi_fastapi_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
REFUSAL_THRESHOLD = 0.35
TOP_K = 10


def run_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 180) -> str:
    command = ["ollama", "run", model, "--no-stream", "--prompt", prompt]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("ollama is not installed or not on PATH. Install ollama and ensure it is available.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()


def normalize_answer(text: str) -> str:
    return text.strip().replace("\n", " ")


def classification_prompt(question: str, context: str) -> str:
    return (
        "Question:\n" + question + "\n\n"
        "Retrieved Context:\n" + context + "\n\n"
        "Classify the question with respect to the retrieved FastAPI documentation context.\n"
        "Respond with exactly one token from: ANSWERABLE, UNANSWERABLE, FALSE_PREMISE.\n"
        "If the question tries to force a false assumption not supported by the docs, answer FALSE_PREMISE.\n"
    )


def render_answer_prompt(question: str, citations: list[dict], context: str) -> str:
    citation_text = "\n".join(
        [f"[{i+1}] {item['section']} - {item['url']}" for i, item in enumerate(citations)]
    )
    return (
        "Use only the FastAPI documentation context below to answer the question.\n"
        "If the answer is not available in the context, say you cannot answer from docs.\n\n"
        "Question:\n" + question + "\n\n"
        "Citations:\n" + citation_text + "\n\n"
        "Context:\n" + context + "\n\n"
        "Answer concisely and include the citations by index if a direct answer is available.\n"
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
    query_results = collection.query(
        query_texts=[question],
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

    top_distance = candidates[0]["distance"]
    if top_distance is None or top_distance > REFUSAL_THRESHOLD:
        return {
            "type": "REFUSAL",
            "message": "Not answerable from docs",
            "citations": [],
            "latency": time.perf_counter() - start_time,
        }

    reranked = rerank_with_cross_encoder(question, candidates)
    chosen = reranked[:5]
    context = "\n\n".join([item["content"] for item in chosen])
    top_context = "\n\n".join([item["content"] for item in chosen[:3]])
    classifier_input = classification_prompt(question, top_context)
    classification = normalize_answer(run_ollama(classifier_input)).upper()
    if "FALSE_PREMISE" in classification:
        return {
            "type": "FALSE_PREMISE",
            "message": "The question contains a false premise not supported by FastAPI docs.",
            "citations": [item["metadata"] for item in chosen[:3]],
            "latency": time.perf_counter() - start_time,
        }
    if "UNANSWERABLE" in classification:
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

if __name__ == "__main__":
    example = "How do query parameters work in FastAPI?"
    response = ask_question(example)
    print(json.dumps(response, indent=2, ensure_ascii=False))
