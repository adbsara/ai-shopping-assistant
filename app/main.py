"""FastAPI service: semantic product search."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.embeddings import Embedder
from app.vector_store import ProductStore

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy objects ONCE at startup, not on every request
    resources["embedder"] = Embedder()
    resources["store"] = ProductStore()
    yield
    resources.clear()


app = FastAPI(
    title="AI Shopping Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Request and response shapes ----------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500,
                       examples=["lightweight backpack for day hiking"])
    top_k: int = Field(5, ge=1, le=20)
    max_price: float | None = Field(None, gt=0)
    min_rating: float | None = Field(None, ge=0, le=5)


class Product(BaseModel):
    title: str
    price: float
    rating: float
    rating_count: int
    category: str
    image: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Product]
    embed_ms: float
    search_ms: float


# ---------- Endpoints ----------

@app.get("/health")
def health():
    """Simple check that the service is alive."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    vector = resources["embedder"].embed_one(req.query)
    t1 = time.perf_counter()
    hits = resources["store"].search(
        vector,
        limit=req.top_k,
        max_price=req.max_price,
        min_rating=req.min_rating,
    )
    t2 = time.perf_counter()

    results = []
    for hit in hits:
        p = hit.payload
        results.append(Product(
            title=p["title"],
            price=p["price"],
            rating=p["rating"],
            rating_count=p["rating_count"],
            category=p["category"],
            image=p.get("image"),
            score=round(hit.score, 4),
        ))

    return SearchResponse(
        query=req.query,
        results=results,
        embed_ms=round((t1 - t0) * 1000, 1),
        search_ms=round((t2 - t1) * 1000, 1),
    )