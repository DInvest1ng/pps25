import asyncio
import io
from datetime import datetime
from datasets import load_dataset
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from PIL import Image as PILImage
from minio import Minio
from src.db.models import Base, Image, Item

DATABASE_URL = "postgresql+asyncpg://myuser:1@localhost:5433/mydb"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "memes"

async def init_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def init_minio():
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
        print(f"Создан MinIO bucket: {MINIO_BUCKET}")
    return client

async def fill_db_from_hf():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await init_db(engine)

    minio_client = init_minio()

    dataset = load_dataset("DIvest1ng/meme", split="train")
    print("Dataset loaded, total records:", len(dataset))

    async with async_session() as session:
        print("Начинаем загрузку и сохранение...")

        for i, item in enumerate(dataset, start=1):
            image = item["image"]
            description = item.get("description", None)

            if isinstance(image, PILImage.Image):
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_bytes.seek(0)

                object_key = f"meme_{i:05d}.jpg"

                minio_client.put_object(
                    bucket_name=MINIO_BUCKET,
                    object_name=object_key,
                    data=image_bytes,
                    length=image_bytes.getbuffer().nbytes,
                    content_type="image/jpeg"
                )

                new_image = Image(
                    object_key=object_key,
                    created_at=datetime.now()
                )
                session.add(new_image)

                new_item = Item(
                    description=description,
                    image_key=object_key,
                    created_at=datetime.now()
                )
                session.add(new_item)

            if i % 500 == 0:
                print(f"{i} images processed and committed...")
                await session.commit()

        await session.commit()

    print("Finished uploading all images and descriptions to DB and MinIO.")



if __name__ == "__main__":
    asyncio.run(fill_db_from_hf())
