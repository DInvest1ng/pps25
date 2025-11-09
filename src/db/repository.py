from src.db.models import Image, Item
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

class Repository:
    @staticmethod
    async def add_image(session: AsyncSession, object_key: str) -> Image:
        img = Image(object_key=object_key)
        session.add(img)
        await session.commit()
        await session.refresh(img)
        return img

    @staticmethod
    async def add_item(session: AsyncSession, description: str, image_key: str) -> Item:
        item = Item(description=description, image_key=image_key)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return item

    @staticmethod
    async def get_item_by_id(session: AsyncSession, item_id: int):
        result = await session.execute(select(Item).where(Item.id == item_id))
        return result.scalars().first()

    @staticmethod
    async def list_items(session: AsyncSession, limit: int = 10):
        result = await session.execute(select(Item).where(Item.is_deleted == False).limit(limit))
        return result.scalars().all()
