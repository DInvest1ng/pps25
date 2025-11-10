import asyncio
from src.db.init_db import SessionLocal
from src.search.vector_search import VectorSearch

async def main():
    async with SessionLocal() as session:
        vs = VectorSearch()
        await vs.build_index(session)
        print(await vs.search(session, "смешной кот"))
        print(await vs.search(session, "грустный кот"))

if __name__ == "__main__":
    asyncio.run(main())
