import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch
from chromadb import Client
from chromadb.config import Settings

DATA_PATH = Path("data/fastapi_docs.json")
CHROMA_DIR = Path("db/chroma")
COLLECTION_NAME = "fastapi_fastapi_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class BgeTextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 8
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
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
                batch_embeddings = (summed / divisor).cpu().tolist()
                embeddings.extend(batch_embeddings)
        return embeddings


def load_documents() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing scraped docs at {DATA_PATH}. Run scrape_fastapi_docs.py first.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("documents", [])


def build_chunks(documents: list[dict]) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = []
    metadatas = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue
        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "url": doc["url"],
                    "section": doc["title"],
                    "chunk_index": idx,
                }
            )
    return texts, metadatas


def build_chroma_collection(texts: list[str], metadatas: list[dict]) -> None:
    embedder = BgeTextEmbedder()
    embeddings = embedder.embed_texts(texts)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(settings=Settings(is_persistent=True, persist_directory=str(CHROMA_DIR)))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    ids = [f"chunk-{i}" for i in range(len(texts))]
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )
    print(f"Saved {len(texts)} chunks to Chroma in {CHROMA_DIR}")


if __name__ == "__main__":
    documents = load_documents()
    texts, metadatas = build_chunks(documents)
    build_chroma_collection(texts, metadatas)
