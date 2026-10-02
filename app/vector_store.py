"""Access layer for Qdrant: create collection, insert products, search."""
from qdrant_client import QdrantClient, models

from app import config


class ProductStore:
    def __init__(self):
        self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.collection = config.COLLECTION

    def recreate(self):
        """Delete the collection if it exists and create a fresh one."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        # Indexes make filtering by these fields fast
        self.client.create_payload_index(self.collection, "price", models.PayloadSchemaType.FLOAT)
        self.client.create_payload_index(self.collection, "rating", models.PayloadSchemaType.FLOAT)
        self.client.create_payload_index(self.collection, "category", models.PayloadSchemaType.KEYWORD)

    def upsert(self, ids, vectors, payloads):
        points = [
            models.PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, limit=5, max_price=None, min_rating=None):
        conditions = []
        if max_price is not None:
            conditions.append(models.FieldCondition(key="price", range=models.Range(lte=max_price)))
        if min_rating is not None:
            conditions.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))

        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        return result.points

    def count(self):
        return self.client.count(self.collection).count