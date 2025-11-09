import asyncio
import os
import math
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sqlalchemy import select, outerjoin
from sqlalchemy.orm import aliased

from src.db.init_db import SessionLocal, init_db
from src.db.models import Item, Image
from src.model.model import CLIPVectorizer
from transformers.image_utils import load_image
from PIL import Image as PILImage

BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", "256"))
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "id_map.npy")
PREFER_IMAGE = os.getenv("PREFER_IMAGE", "1") == "1"

def try_load_image(key: str) -> Optional[PILImage.Image]:
    if not key:
        return None
    try:
        img = load_image(key)
        return img
    except Exception:
        try:
            if os.path.exists(key):
                return PILImage.open(key).convert("RGB")
        except Exception:
            return None
    return None

async def fetch_batch(session, offset: int, limit: int) -> List[Tuple[int, Optional[str], Optional[str]]]:
    img_alias = aliased(Image)
    stmt = (
        select(Item.id, Item.description, img_alias.object_key)
        .select_from(Item)
        .outerjoin(img_alias, Item.image_key == img_alias.object_key)
        .where(Item.is_deleted == False)
        .order_by(Item.id)
        .offset(offset)
        .limit(limit)
    )
    q = await session.execute(stmt)
    return q.all()

async def build_and_save_index(batch_size: int = BATCH_SIZE, prefer_image: bool = PREFER_IMAGE):
    await init_db()
    async with SessionLocal() as session:
        total_q = await session.execute(select(Item).where(Item.is_deleted == False))
        total_count = len(total_q.scalars().all())
        if total_count == 0:
            print("no items found")
            return

        vectorizer = CLIPVectorizer()
        faiss_index = None
        id_map: List[int] = []
        processed = 0
        offset = 0
        while processed < total_count:
            rows = await fetch_batch(session, offset=offset, limit=batch_size)
            if not rows:
                break
            offset += len(rows)
            processed += len(rows)

            ids = [r[0] for r in rows]
            descs = [r[1] or "" for r in rows]
            img_keys = [r[2] for r in rows]

            images = []
            image_positions = []
            text_positions = []
            texts_for_text = []

            if prefer_image:
                for i, key in enumerate(img_keys):
                    img = try_load_image(key)
                    if img is not None:
                        images.append(img)
                        image_positions.append(i)
                    else:
                        text_positions.append(i)
                        texts_for_text.append(descs[i])
            else:
                texts_for_text = descs
                text_positions = list(range(len(descs)))

            embeddings_batch = None
            dim = None

            if images:
                img_emb = vectorizer.encode_image(images, batch_size=32, normalize=True)
                dim = img_emb.shape[1]
                embeddings_batch = np.zeros((len(rows), dim), dtype=np.float32)
                for pos_idx, emb in zip(image_positions, img_emb):
                    embeddings_batch[pos_idx] = emb

            if text_positions:
                txt_emb = vectorizer.encode_text(texts_for_text, batch_size=128, normalize=True)
                if dim is None:
                    dim = txt_emb.shape[1]
                    embeddings_batch = np.zeros((len(rows), dim), dtype=np.float32)
                for pos_idx, emb in zip(text_positions, txt_emb):
                    embeddings_batch[pos_idx] = emb

            if embeddings_batch is None:
                continue

            if faiss_index is None:
                faiss_index = faiss.IndexFlatIP(dim)

            faiss.normalize_L2(embeddings_batch)
            faiss_index.add(embeddings_batch.astype(np.float32))
            id_map.extend(ids)

            print(f"processed {processed}/{total_count}")

        if faiss_index is None:
            print("no embeddings were created")
            return

        faiss.write_index(faiss_index, INDEX_PATH)
        np.save(ID_MAP_PATH, np.array(id_map, dtype=np.int64))
        print("saved index to", INDEX_PATH)
        print("saved id map to", ID_MAP_PATH)

def main():
    asyncio.run(build_and_save_index())

if __name__ == "__main__":
    main()
