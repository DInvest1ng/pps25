import asyncio
from datasets import load_dataset
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import Base, Image
from PIL import Image as PILImage
import io

DATABASE_URL = "postgresql+asyncpg://myuser:1@localhost:5433/mydb"

async def init_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def fill_db_from_hf():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await init_db(engine)

    dataset = load_dataset("DIvest1ng/meme", split="train") # train[:100]
    async with async_session() as session:
        for i, item in enumerate(dataset):
            image = item["image"]
            if isinstance(image, PILImage.Image):
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                object_key = f"meme_{i}.jpg"
                new_image = Image(object_key=object_key, created_at=datetime.now())
                session.add(new_image)
        await session.commit()

if __name__ == "__main__":
    asyncio.run(fill_db_from_hf())
