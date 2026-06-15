import argparse
import json
import re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://fastapi.tiangolo.com"
OUTPUT_PATH = Path("data/fastapi_docs.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TruthGateBot/1.0; +https://github.com)"
}

SKIP_PATTERNS = [".pdf", "mailto:", "tel:"]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    normalized = urljoin(BASE_URL, path)
    return normalized


def is_valid_link(url: str) -> bool:
    if any(pattern in url for pattern in SKIP_PATTERNS):
        return False
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    return True


def extract_page_document(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    main = soup.find("main") or soup.find("article") or soup.body

    if main is None:
        content = soup.get_text(" ", strip=True)
    else:
        paragraphs = []
        for element in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "code", "blockquote"]):
            text = element.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
        content = "\n".join(paragraphs)

    content = re.sub(r"\s+", " ", content).strip()
    return {
        "title": title,
        "url": url,
        "content": content,
    }


def gather_urls(max_pages: int | None = None) -> list[str]:
    visited = set()
    queue = deque([BASE_URL + "/"])
    discovered = []

    while queue:
        if max_pages is not None and len(visited) >= max_pages:
            break
        current_url = queue.popleft()
        canonical = normalize_url(current_url)
        if canonical in visited:
            continue
        visited.add(canonical)

        try:
            response = requests.get(canonical, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            print(f"Skipping {canonical}: {exc}")
            continue

        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        discovered.append(canonical)
        print(f"Found page: {canonical} (queue={len(queue)})")

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href.startswith("#"):
                continue
            next_url = urljoin(canonical, href)
            if not is_valid_link(next_url):
                continue
            normalized = normalize_url(next_url)
            if normalized not in visited:
                queue.append(normalized)

    return sorted(set(discovered))


def scrape_all(max_pages: int | None = None) -> list[dict]:
    urls = gather_urls(max_pages=max_pages)
    docs = []
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            document = extract_page_document(url, response.text)
            if document["content"]:
                docs.append(document)
                print(f"Scraped {url} -> {len(document['content'])} chars")
        except Exception as exc:
            print(f"Failed to scrape {url}: {exc}")
    return docs


def save_output(docs: list[dict]) -> None:
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"source": BASE_URL, "count": len(docs), "documents": docs}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(docs)} docs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FastAPI documentation.")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to crawl")
    args = parser.parse_args()
    docs = scrape_all(max_pages=args.max_pages)
    save_output(docs)
