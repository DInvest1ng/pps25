import os
import io
import json
import logging
import asyncio
import random
import re
from typing import Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup
from aiogram.filters import Command, CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from minio import Minio
from minio.error import S3Error

from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in environment")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL in environment")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "memes")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

MEME_MAX_INDEX_ENV = os.getenv("MEME_MAX_INDEX")
RANDOM_TRIES = int(os.getenv("RANDOM_TRIES", "12"))

FAVORITES_FILE = os.getenv("FAVORITES_FILE", "favorites.json")
FILE_IDS_FILE = os.getenv("FILE_IDS_FILE", "file_ids.json")

bot = Bot(token=TOKEN)
dp = Dispatcher()

engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionMaker = async_sessionmaker(engine, expire_on_commit=False)

vs: VectorSearch = VectorSearch()

minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

def load_json_file(path: str, default=None):
    if default is None:
        default = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load JSON file %s", path)
    return default

def save_json_file(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save JSON file %s", path)

favorites: Dict[str, List[int]] = load_json_file(FAVORITES_FILE, {})
file_id_cache: Dict[str, str] = load_json_file(FILE_IDS_FILE, {})

MAX_MEME_INDEX: Optional[int] = None
MEME_INDEX_PATTERN = re.compile(r"meme_(\d{5})\.jpg$", re.IGNORECASE)

async def minio_object_exists(bucket: str, object_name: str) -> bool:
    def _stat():
        try:
            minio_client.stat_object(bucket, object_name)
            return True
        except Exception:
            return False
    return await asyncio.to_thread(_stat)

async def download_object_bytes(bucket: str, object_name: str) -> bytes:
    def _get():
        resp = minio_client.get_object(bucket, object_name)
        try:
            data = resp.read()
            return data
        finally:
            try:
                resp.close()
                resp.release_conn()
            except Exception:
                pass
    return await asyncio.to_thread(_get)

async def init_memes_index():
    global MAX_MEME_INDEX
    if MEME_MAX_INDEX_ENV:
        try:
            MAX_MEME_INDEX = int(MEME_MAX_INDEX_ENV)
            logger.info("Using MEME_MAX_INDEX from env: %s", MAX_MEME_INDEX)
            return
        except Exception:
            logger.exception("Invalid MEME_MAX_INDEX env, will try to scan MinIO")

    logger.info("Scanning MinIO bucket '%s' for meme_*.jpg objects to detect max index...", MINIO_BUCKET)

    def _list_and_find_max():
        max_idx = 0
        try:
            objs = minio_client.list_objects(MINIO_BUCKET, prefix="", recursive=True)
            for o in objs:
                name = o.object_name
                m = MEME_INDEX_PATTERN.search(name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except Exception:
                        pass
            return max_idx
        except Exception as e:
            logger.exception("Failed to list objects in MinIO: %s", e)
            return 0

    max_found = await asyncio.to_thread(_list_and_find_max)
    if max_found > 0:
        MAX_MEME_INDEX = max_found
        logger.info("Detected MAX_MEME_INDEX = %s", MAX_MEME_INDEX)
    else:
        MAX_MEME_INDEX = 0
        logger.warning("No meme_*.jpg objects found in MinIO (or listing failed). Set MAX_MEME_INDEX=0")

def create_main_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🎲 Случайный мем", callback_data="random")
    kb.button(text="🔍 Поиск мемов", callback_data="search")
    kb.button(text="❤️ Мои избранные", callback_data="my_favorites")
    kb.button(text="❓ Помощь", callback_data="help")
    kb.adjust(2, 2)
    return kb.as_markup()

def create_meme_keyboard(meme_idx: int, query: str = "") -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if query:
        kb.button(text="🔁 Ещё по этому запросу", callback_data=f"more:{query}")
    kb.button(text="❤️ В избранное", callback_data=f"fav:{meme_idx}")
    kb.button(text="🎲 Случайный мем", callback_data="random")
    kb.button(text="🏠 В меню", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()

def create_favorites_keyboard(meme_idx: int, current_page: int = 0) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🗑️ Удалить из избранного", callback_data=f"unfav:{meme_idx}")
    kb.button(text="⬅️ Предыдущий", callback_data=f"fav_prev:{current_page - 1}")
    kb.button(text="➡️ Следующий", callback_data=f"fav_next:{current_page + 1}")
    kb.button(text="🏠 В меню", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()

async def send_meme_by_object(chat_id: int, item_id: int, object_name: str,
                              reply_markup: Optional[InlineKeyboardMarkup] = None,
                              caption: Optional[str] = None):
    key = str(item_id)
    file_id = file_id_cache.get(key)
    if file_id:
        try:
            await bot.send_photo(chat_id=chat_id, photo=file_id, reply_markup=reply_markup, caption=caption or "")
            return
        except Exception:
            logger.exception("Failed to send cached file_id, will re-upload for item %s", item_id)
            file_id_cache.pop(key, None)
            save_json_file(FILE_IDS_FILE, file_id_cache)

    try:
        data = await download_object_bytes(MINIO_BUCKET, object_name)
    except S3Error as se:
        logger.warning("MinIO S3Error for %s: %s", object_name, se)
        await bot.send_message(chat_id, f"Файл {object_name} отсутствует в хранилище.")
        return
    except Exception:
        logger.exception("Failed to download %s from MinIO", object_name)
        await bot.send_message(chat_id, "Ошибка при загрузке файла из хранилища.")
        return

    bio = io.BytesIO(data)
    bio.name = object_name
    bio.seek(0)

    input_file = types.InputFile(bio, filename=object_name)

    try:
        msg = await bot.send_photo(chat_id=chat_id, photo=input_file, reply_markup=reply_markup, caption=caption or "")
        try:
            fid = msg.photo[-1].file_id
            file_id_cache[key] = fid
            save_json_file(FILE_IDS_FILE, file_id_cache)
        except Exception:
            logger.exception("Can't cache file_id for item %s", item_id)
    except Exception:
        logger.exception("Failed to send meme %s to chat %s", object_name, chat_id)
        await bot.send_message(chat_id, "Не удалось отправить изображение.")


async def choose_random_meme_object_name(tries: int = RANDOM_TRIES) -> Optional[Tuple[int, str]]:
    global MAX_MEME_INDEX
    if not MAX_MEME_INDEX or MAX_MEME_INDEX <= 0:
        logger.warning("MAX_MEME_INDEX is not set or zero, can't choose random meme")
        return None

    for _ in range(tries):
        i = random.randint(1, MAX_MEME_INDEX)
        object_name = f"meme_{i:05d}.jpg"
        if await minio_object_exists(MINIO_BUCKET, object_name):
            return i, object_name
    return None

async def startup_load_index():
    logger.info("Starting to build vector index (one-time)...")
    async with AsyncSessionMaker() as session:
        try:
            await vs.build_index(session)
            logger.info("Vector index built successfully.")
        except Exception:
            logger.exception("Failed to build vector index")

async def perform_search(chat_id: int, query: str, user_id: int):
    await bot.send_chat_action(chat_id, "typing")
    try:
        async with AsyncSessionMaker() as session:
            ids: List[int] = await vs.search(session, query)
    except Exception:
        logger.exception("Vector search failed")
        await bot.send_message(chat_id, "Ошибка при поиске — попробуйте позже.")
        return

    if not ids:
        await bot.send_message(chat_id, "По запросу ничего не найдено.")
        return

    ids = ids[:10]
    first = ids[0]
    markup = create_meme_keyboard(first, query if len(ids) > 1 else "")

    object_name = f"meme_{first:05d}.jpg"
    try:
        async with AsyncSessionMaker() as session:
            res = await session.execute(text("SELECT images.object_key FROM items JOIN images ON items.image_key = images.object_key WHERE items.id = :id LIMIT 1").bindparams(id=first))
            row = res.first()
            if row and row[0]:
                raw = row[0]
                if raw.startswith(f"{MINIO_BUCKET}/"):
                    object_name = raw.split("/", 1)[1]
                else:
                    object_name = raw
    except Exception:
        logger.debug("Failed to resolve image_key from DB for id %s, fallback to conventional name", first)

    await send_meme_by_object(chat_id, first, object_name, reply_markup=markup)
    if len(ids) > 1:
        await bot.send_message(chat_id, f"Найдено {len(ids)} результатов. Нажмите «Ещё по этому запросу» чтобы увидеть больше.", reply_markup=None)

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    first_name = message.from_user.first_name or "пользователь"
    txt = (
        f"Привет, {first_name}!\n\n"
        "Я — бот для поиска мемов.\n"
        "Использование:\n"
        " • Нажмите «🎲 Случайный мем» чтобы получить случайный мем.\n"
        " • Напишите текст в чат — бот выполнит поиск и пришлёт подходящий мем.\n"
        " • Ставьте ❤️ чтобы добавить мем в избранное.\n"
    )
    await message.answer(txt, reply_markup=create_main_keyboard())

@dp.message(Command(commands=["random"]))
async def cmd_random(message: types.Message):
    await handle_send_random(message.chat.id, message.from_user.id)

@dp.message(Command(commands=["favorites"]))
async def cmd_favorites(message: types.Message):
    await show_favorite_meme(message.chat.id, message.from_user.id, 0)

@dp.message(Command(commands=["help"]))
async def cmd_help(message: types.Message):
    await message.answer("Помощь: используйте кнопки меню или просто отправьте текст для поиска мемов.", reply_markup=create_main_keyboard())

@dp.message()
async def handle_text(message: types.Message):
    query = (message.text or "").strip()
    if not query:
        return
    if len(query) < 2:
        await message.answer("Введите как минимум 2 символа для поиска.")
        return
    await perform_search(message.chat.id, query, message.from_user.id)

@dp.callback_query()
async def handle_callbacks(callback: types.CallbackQuery):
    data = callback.data or ""
    uid = str(callback.from_user.id)
    try:
        if data == "random":
            await callback.answer("🎲 Ищу случайный мем...")
            await handle_send_random(callback.message.chat.id, callback.from_user.id)

        elif data == "search":
            await callback.answer()
            await callback.message.answer("Введи запрос для поиска мемов 🔍")

        elif data == "my_favorites":
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, callback.from_user.id, 0)

        elif data.startswith("fav:"):
            meme_idx = int(data.split(":", 1)[1])
            user_favs = set(favorites.get(uid, []))
            if meme_idx in user_favs:
                await callback.answer("Уже в избранном")
            else:
                user_favs.add(meme_idx)
                favorites[uid] = list(user_favs)
                save_json_file(FAVORITES_FILE, favorites)
                await callback.answer("❤️ Добавлено в избранное")

        elif data.startswith("unfav:"):
            meme_idx = int(data.split(":", 1)[1])
            user_favs = favorites.get(uid, [])
            if meme_idx in user_favs:
                user_favs.remove(meme_idx)
                favorites[uid] = user_favs
                save_json_file(FAVORITES_FILE, favorites)
                await callback.answer("🗑️ Удалено из избранного")
                try:
                    await callback.message.delete()
                except Exception:
                    pass
                if favorites[uid]:
                    await show_favorite_meme(callback.message.chat.id, callback.from_user.id, 0)
                else:
                    await bot.send_message(callback.message.chat.id, "❤️ Избранные пусты", reply_markup=create_main_keyboard())
            else:
                await callback.answer("Этого мема нет в избранном")

        elif data.startswith("fav_next:") or data.startswith("fav_prev:"):
            page = int(data.split(":", 1)[1])
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, callback.from_user.id, page)

        elif data.startswith("more:"):
            query = data.split(":", 1)[1]
            await callback.answer("🔍 Ищу ещё мемы...")
            await perform_search(callback.message.chat.id, query, callback.from_user.id)

        elif data == "menu":
            await callback.answer()
            await cmd_start(callback.message)

        elif data == "help":
            await callback.answer()
            await cmd_help(callback.message)

        else:
            await callback.answer()

    except Exception:
        logger.exception("Error in callback handler")
        try:
            await callback.answer("Произошла ошибка")
        except Exception:
            pass

async def show_favorite_meme(chat_id: int, user_id: int, page: int = 0):
    user_favs = favorites.get(str(user_id), [])
    if not user_favs:
        await bot.send_message(chat_id, "❤️ У вас пока нет избранных мемов", reply_markup=create_main_keyboard())
        return

    if page < 0:
        page = 0
    if page >= len(user_favs):
        page = len(user_favs) - 1

    idx = user_favs[page]
    caption = f"📖 {page + 1}/{len(user_favs)}"

    object_name = f"meme_{idx:05d}.jpg"
    await send_meme_by_object(chat_id, idx, object_name, reply_markup=create_favorites_keyboard(idx, page), caption=caption)

async def handle_send_random(chat_id: int, user_id: int):
    try:
        res = await choose_random_meme_object_name()
        if not res:
            await bot.send_message(chat_id, "В базе нет мемов или объекты отсутствуют в MinIO.")
            return
        index, object_name = res
        await send_meme_by_object(chat_id, index, object_name, reply_markup=create_meme_keyboard(index, ""))
    except Exception:
        logger.exception("Failed to send random meme")
        await bot.send_message(chat_id, "Ошибка при получении случайного мема.")

async def main():
    logger.info("Bot starting...")
    await init_memes_index()

    await startup_load_index()

    try:
        await bot.set_my_commands([
            types.BotCommand(command="/start", description="Запуск бота"),
            types.BotCommand(command="/random", description="Случайный мем"),
            types.BotCommand(command="/favorites", description="Мои избранные"),
            types.BotCommand(command="/help", description="Помощь"),
        ])
    except Exception:
        logger.exception("Failed to set bot commands")

    try:
        await dp.start_polling(bot)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
