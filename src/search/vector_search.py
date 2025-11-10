import os
import faiss
import torch
import numpy as np
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.init_db import SessionLocal
from sqlalchemy import select
from src.db.models import Item, Image
from src.model.model import CLIPVectorizer

class VectorSearch:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device = None,
        index_path: str = "data/faiss_index.bin",
        id_map_path: str = "data/id_map.npy",
    ):
        self.model = CLIPVectorizer(model_name=model_name, device=device)
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.index = None
        self.id_map = None

        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path):
            self.load_index()

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        self.id_map = np.load(self.id_map_path)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def save_index(self):
        if self.index is not None and self.id_map is not None:
            faiss.write_index(self.index, self.index_path)
            np.save(self.id_map_path, self.id_map)
            print("Index and ID map saved.")

    async def build_index(self, session: AsyncSession, limit = None):
        q = select(Item.id, Item.description).where(Item.is_deleted == False).order_by(Item.id)
        if limit:
            q = q.limit(limit)
        res = await session.execute(q)
        rows = res.all()

        if not rows:
            print("No data found to build index.")
            return

        ids = [r[0] for r in rows]
        texts = [r[1] or "" for r in rows]
        embeddings = self.model.encode_text(texts)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.id_map = np.array(ids, dtype=np.int64)

        self.save_index()

    async def search(self, session: AsyncSession, query_text: str, top_k: int = 30) -> List[int]:
        if self.index is None or self.id_map is None:
            print("Index not loaded — building from DB...")
            await self.build_index(session)

        query_vec = self.model.encode_text(query_text)
        faiss.normalize_L2(query_vec)

        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        indices = I[0]
        result_ids = [int(self.id_map[i]) for i in indices if i < len(self.id_map)]
        return result_ids


import asyncio
async def main():
    async with SessionLocal() as session:
        vs = VectorSearch()
        await vs.build_index(session)
        result = await vs.search(session, "смешной кот")
        print("Top results:", result)

if __name__ == "__main__":
    asyncio.run(main())