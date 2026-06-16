import chromadb

client = chromadb.PersistentClient(path="db/chroma")

collection = client.get_collection("fastapi_fastapi_docs")

print("Count:", collection.count())

result = collection.peek(limit=3)

print(result)


# there is a lot of repitition in the retrieved chunks need to do smthg
# may be hue to during scraping it also contains the navigatin menu, table of contents and all
# since this is the initial ingestion, we can go with it