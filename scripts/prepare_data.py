"""
Prepare a clean product catalog from the Amazon Reviews 2023 dataset
(Sports & Outdoors category).

Run:
    python -m scripts.prepare_data --limit 1000
"""
import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfFileSystem

HF_PATH = (
    "datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/"
    "meta_Sports_and_Outdoors.jsonl"
)
OUT_FILE = Path("data/products.jsonl")


def parse_price(value):
    """Convert price values like 12.5, '$12.50' or 'None' to a float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def clean(raw, min_reviews):
    """Keep only useful fields; return None for low-quality products."""
    title = (raw.get("title") or "").strip()
    price = parse_price(raw.get("price"))
    reviews = raw.get("rating_number") or 0
    if not title or price is None or reviews < min_reviews:
        return None

    categories = raw.get("categories") or []
    category = categories[1] if len(categories) > 1 else (raw.get("main_category") or "Other")
    category_path = " > ".join(categories) if categories else category
    features = (raw.get("features") or [])[:6]
    description = " ".join(raw.get("description") or [])[:800]
    images = raw.get("images") or []
    image = (images[0] or {}).get("large") if images else None

    # The text we will turn into a vector: the most meaningful info in one string
    text_parts = [title, f"Category: {category_path}"]
    if features:
        text_parts.append("Features: " + " ".join(features))
    if description:
        text_parts.append("Description: " + description)

    return {
        "asin": raw.get("parent_asin"),
        "title": title,
        "price": price,
        "rating": float(raw.get("average_rating") or 0),
        "rating_count": int(reviews),
        "store": raw.get("store"),
        "category": category,
        "category_path": category_path,
        "features": features,
        "description": description,
        "image": image,
        "text": "\n".join(text_parts)[:2000],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-reviews", type=int, default=20)
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(exist_ok=True)
    kept = scanned = 0

    # Stream the file line by line instead of downloading several GB
    with HfFileSystem().open(HF_PATH, "r", encoding="utf-8") as source, \
         OUT_FILE.open("w", encoding="utf-8") as out:
        for line in source:
            scanned += 1
            try:
                item = clean(json.loads(line), args.min_reviews)
            except json.JSONDecodeError:
                continue
            if item is None:
                continue
            item["id"] = kept
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 200 == 0:
                print(f"kept {kept} / scanned {scanned}")
            if kept >= args.limit:
                break

    print(f"Done: {kept} products saved to {OUT_FILE} (scanned {scanned}).")


if __name__ == "__main__":
    main()