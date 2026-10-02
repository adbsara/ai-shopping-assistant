"""Turn text into vectors using the multilingual bge-m3 model."""
from sentence_transformers import SentenceTransformer

from app import config


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        # Longer texts are cut to 512 tokens: faster, and enough for product info
        self.model.max_seq_length = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=16)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]