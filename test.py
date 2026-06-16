import json

with open("data/fastapi_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

for doc in docs["documents"][:10]:
    print(doc["title"])
    print(doc["url"])
    print("-" * 50)