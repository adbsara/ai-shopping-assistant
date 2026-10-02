"""
Embed all products and store them in Qdrant.

Run:
    python -m scripts.index_products
"""
import json
from pathlib import Path

from tqdm import tqdm

from app.embeddings import Embedder
from app.vector_store import ProductStore

DATA_FILE = Path("data/products.jsonl")
BATCH_SIZE = 32


def main():
    lines = DATA_FILE.read_text(encoding="utf-8").splitlines()
    products = [json.loads(line) for line in lines if line.strip()]
    print(f"Loaded {len(products)} products")

    print("Loading embedding model (first time downloads ~2 GB)...")
    embedder = Embedder()

    store = ProductStore()
    store.recreate()

    for start in tqdm(range(0, len(products), BATCH_SIZE), desc="Indexing"):
        batch = products[start:start + BATCH_SIZE]
        vectors = embedder.embed([p["text"] for p in batch])
        # Everything except id and text is stored as payload
        payloads = [{k: v for k, v in p.items() if k not in ("id", "text")} for p in batch]
        store.upsert([p["id"] for p in batch], vectors, payloads)

    print(f"Done. Products in Qdrant: {store.count()}")


if __name__ == "__main__":
    main()