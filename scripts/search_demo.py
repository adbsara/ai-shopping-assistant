"""
Test semantic search from the command line.

Run:
    python -m scripts.search_demo "lightweight backpack for day hiking" --max-price 80
"""
import argparse
import time

from app.embeddings import Embedder
from app.vector_store import ProductStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    args = parser.parse_args()

    embedder = Embedder()
    store = ProductStore()

    t0 = time.perf_counter()
    vector = embedder.embed_one(args.query)
    t1 = time.perf_counter()
    hits = store.search(vector, limit=args.top_k,
                        max_price=args.max_price, min_rating=args.min_rating)
    t2 = time.perf_counter()

    print(f"\nembed: {(t1 - t0) * 1000:.0f} ms | search: {(t2 - t1) * 1000:.0f} ms\n")
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        print(f"{rank}. [{hit.score:.3f}] {p['title'][:90]}")
        print(f"   ${p['price']:.2f} | rating {p['rating']} ({p['rating_count']}) | {p['category']}\n")


if __name__ == "__main__":
    main()