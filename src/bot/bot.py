import os
import logging
import asyncio
import json
import base64
from typing import List, Optional
import html

from datasets import load_dataset
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    InlineQueryResultPhoto,
    InlineQueryResultArticle,
    InputTextMessageContent,
    BufferedInputFile
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties


load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN", "8491826572:AAEH4n6VT64rusidFKEF43Ciii6SCujVDPk")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("memebot")

df = None
try:
    logger.info("Загрузка датасета с HuggingFace...")
    TEST_MODE = False
    SAMPLE_SIZE = 0
    if TEST_MODE:
        logger.info(f"🔬 ТЕСТОВЫЙ РЕЖИМ: загружаем {SAMPLE_SIZE} мемов")
        dataset = load_dataset("DIvest1ng/meme", split=f'train[:{SAMPLE_SIZE}]')
    else:
        logger.info("🚀 ПРОД РЕЖИМ: загружаем все мемы")
        dataset = load_dataset("DIvest1ng/meme", split='train')
    df = dataset.to_pandas()
    logger.info(f"✅ Загружено {len(df)} мемов")
    df['search_text'] = (
            df['description'].fillna('') + ' ' +
            df['alt'].fillna('')
    ).str.lower()

except Exception as e:
    logger.exception("❌ Ошибка загрузки датасета")
    raise

FAVORITES_FILE = "favorites.json"

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Ошибка загрузки favorites.json")
    return {}

def save_favorites(data):
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Ошибка сохранения favorites.json")

favorites = load_favorites()
user_query_history = {}

def get_image_bytes(row) -> bytes:
    try:
        image_data = row['image']
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            if isinstance(image_bytes, bytes):
                return image_bytes
            else:
                logger.error(f"❌ bytes не является bytes, а: {type(image_bytes)}")
                raise ValueError("Invalid bytes format")
        else:
            logger.error(f"❌ Неизвестный формат изображения: {type(image_data)}")
            raise ValueError(f"Unknown image format: {type(image_data)}")

    except Exception as e:
        logger.error(f"❌ Ошибка извлечения изображения для мема #{row.name}: {e}")
        #Заглушка
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def create_input_file(row) -> BufferedInputFile:
    try:
        image_bytes = get_image_bytes(row)
        return BufferedInputFile(image_bytes, filename=f"meme_{row.name}.jpg")
    except Exception as e:
        logger.error(f"❌ Ошибка создания InputFile для мема #{row.name}: {e}")
        # Заглушка в случае ошибки
        placeholder_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        return BufferedInputFile(placeholder_bytes, filename="error.jpg")


def search_memes(query: str, user_id: int, limit: int = 10) -> List[int]:
    if df is None or len(df) == 0:
        return []
    query_clean = query.lower().strip()
    if not query_clean or len(query_clean) < 2:
        return df.sample(min(limit, len(df))).index.tolist()
    query_words = query_clean.split()
    scored = []
    for idx, row in df.iterrows():
        score = 0
        search_text = row['search_text']
        if query_clean in search_text:
            score += 10
        for word in query_words:
            if word in search_text:
                score += 3
        if score > 0:
            scored.append((idx, score))

    if not scored:
        return df.sample(min(limit, len(df))).index.tolist()

    scored.sort(key=lambda x: x[1], reverse=True)
    scored_indices = [idx for idx, _ in scored]
    user_history = user_query_history.get(user_id, {})
    shown_for_query = set(user_history.get(query_clean, []))
    available_indices = [idx for idx in scored_indices if idx not in shown_for_query]
    if not available_indices:
        available_indices = scored_indices
        shown_for_query = set()
    result_indices = available_indices[:limit]
    if user_id not in user_query_history:
        user_query_history[user_id] = {}

    user_query_history[user_id][query_clean] = list(shown_for_query.union(result_indices))
    return result_indices


def create_meme_keyboard(meme_idx: int, query: str = "") -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if query:
        kb.button(text="🔁 Ещё по этому запросу", callback_data=f"more:{query}")

    kb.button(text="❤️ В избранное", callback_data=f"fav:{meme_idx}")
    kb.button(text="🎲 Случайный мем", callback_data="random")
    kb.button(text="🏠 В меню", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()


def create_favorites_keyboard(meme_idx: int, current_page: int = 0) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()

    kb.button(text="🗑️ Удалить из избранного", callback_data=f"unfav:{meme_idx}")
    kb.button(text="⬅️ Предыдущий", callback_data=f"fav_prev:{current_page - 1}")
    kb.button(text="➡️ Следующий", callback_data=f"fav_next:{current_page + 1}")
    kb.button(text="🏠 В меню", callback_data="menu")

    kb.adjust(1, 2, 1)
    return kb.as_markup()

def create_main_keyboard() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🎲 Случайный мем", callback_data="random")
    kb.button(text="❤️ Мои избранные", callback_data="my_favorites")
    kb.button(text="🔍 Поиск мемов", callback_data="search")
    kb.button(text="❓ Помощь", callback_data="help")
    kb.adjust(2, 2)
    return kb.as_markup()


def setup_main_menu():
    main_menu_commands = [
        types.BotCommand(command="/start", description="Перезапустить бота"),
        types.BotCommand(command="/random", description="🎲 Случайный мем"),
        types.BotCommand(command="/search", description="🔍 Поиск мемов"),
        types.BotCommand(command="/favorites", description="❤️ Мои избранные"),
        types.BotCommand(command="/help", description="❓ Помощь")
    ]
    return main_menu_commands


async def show_favorite_meme(chat_id: int, user_id: int, page: int = 0):
    user_favs = favorites.get(str(user_id), [])

    if not user_favs:
        await bot.send_message(chat_id, "❤️ У вас пока нет избранных мемов")
        return

    if page < 0:
        page = 0
    if page >= len(user_favs):
        page = len(user_favs) - 1

    try:
        idx = user_favs[page]
        row = df.iloc[idx]
        input_file = create_input_file(row)

        caption = f"📖 {page + 1}/{len(user_favs)}"

        await bot.send_photo(
            chat_id=chat_id,
            photo=input_file,
            caption=caption,
            reply_markup=create_favorites_keyboard(idx, page)
        )

    except Exception as e:
        logger.error(f"Ошибка показа избранного мема #{idx}: {e}")
        await bot.send_message(chat_id, "❌ Не удалось загрузить избранный мем")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    welcome_text = (
        "🎭 <b>Добро пожаловать в MemeBot!</b>\n\n"
        "Выбери действие ниже 👇"
    )

    await message.answer(welcome_text, reply_markup=create_main_keyboard())


@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    await message.answer("Введи запрос для поиска мемов 🔍")


@dp.message(Command("random"))
async def cmd_random(message: types.Message):
    await send_random_meme(message.chat.id, message.from_user.id)


@dp.message(Command("favorites"))
async def cmd_favorites(message: types.Message):
    user_favs = favorites.get(str(message.from_user.id), [])
    if not user_favs:
        await message.answer("❤️ У вас пока нет избранных мемов")
        return
    await show_favorite_meme(message.chat.id, message.from_user.id, 0)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "🎭 <b>MemeBot - помощь</b>\n\n"
        "<b>Основные команды:</b>\n"
        "• /start - перезапустить бота\n"
        "• /random - случайный мем\n"
        "• /search - поиск мемов\n"
        "• /favorites - мои избранные\n\n"
        "<b>Быстрый доступ:</b>\n"
        "Используй кнопки меню под сообщениями!"
    )

    await message.answer(help_text, reply_markup=create_main_keyboard())


@dp.message()
async def handle_text(message: types.Message):
    query = message.text.strip()
    if len(query) < 2:
        await message.answer("🔍 Введи минимум 2 символа для поиска")
        return

    await perform_search(message.chat.id, query, message.from_user.id)


@dp.inline_query()
async def inline_search(inline_query: types.InlineQuery):
    results = [
        InlineQueryResultArticle(
            id="info",
            title="🎭 MemeBot",
            description="Используйте команды в чате с ботом",
            input_message_content=InputTextMessageContent(
                message_text="🎭 Используйте команды в чате с @mem_ass_bot ботом"
            )
        )
    ]
    await inline_query.answer(results, cache_time=300)


async def send_random_meme(chat_id: int, user_id: int):
    try:
        if df is None or len(df) == 0:
            await bot.send_message(chat_id, "❌ База мемов не загружена")
            return

        row = df.sample(1).iloc[0]
        idx = row.name
        input_file = create_input_file(row)

        await bot.send_photo(
            chat_id=chat_id,
            photo=input_file,
            reply_markup=create_meme_keyboard(idx, "")
        )
        logger.info(f"✅ Отправлен случайный мем #{idx}")

    except Exception as e:
        logger.exception("Ошибка отправки случайного мема")
        await bot.send_message(chat_id, "❌ Ошибка при загрузке мема")


async def perform_search(chat_id: int, query: str, user_id: int):
    try:
        if df is None or len(df) == 0:
            await bot.send_message(chat_id, "❌ База мемов не загружена")
            return

        meme_indices = search_memes(query, user_id, limit=5)
        if not meme_indices:
            await bot.send_message(chat_id, f"❌ По запросу «{query}» ничего не найдено")
            return

        row = df.iloc[meme_indices[0]]
        idx = meme_indices[0]
        input_file = create_input_file(row)
        await bot.send_photo(
            chat_id=chat_id,
            photo=input_file,
            reply_markup=create_meme_keyboard(idx, query)
        )
        logger.info(f"✅ Отправлен мем #{idx} по запросу '{query}'")
        if len(meme_indices) > 1:
            await bot.send_message(
                chat_id,
                f"Нажми 'Ещё по этому запросу' чтобы увидеть больше мемов на эту тематику"
            )

    except Exception as e:
        logger.exception("Ошибка поиска")
        await bot.send_message(chat_id, "❌ Ошибка при поиске мемов")


@dp.callback_query()
async def handle_callbacks(callback: types.CallbackQuery):
    data = callback.data
    user_id = str(callback.from_user.id)
    try:
        if data == "random":
            await callback.answer("🎲 Ищу случайный мем...")
            await send_random_meme(callback.message.chat.id, callback.from_user.id)

        elif data == "my_favorites":
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, callback.from_user.id, 0)

        elif data.startswith("fav_next:"):
            page = int(data.split(":")[1])
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, callback.from_user.id, page)

        elif data.startswith("fav_prev:"):
            page = int(data.split(":")[1])
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, callback.from_user.id, page)

        elif data.startswith("unfav:"):
            meme_idx = int(data.split(":")[1])
            user_favs = favorites.get(user_id, [])

            if meme_idx in user_favs:
                user_favs.remove(meme_idx)
                favorites[user_id] = user_favs
                save_favorites(favorites)
                await callback.answer("🗑️ Удалено из избранного")
                await callback.message.delete()
                if user_favs:
                    await show_favorite_meme(callback.message.chat.id, callback.from_user.id, 0)
                else:
                    await callback.message.answer("❤️ Избранные мемы очищены", reply_markup=create_main_keyboard())
            else:
                await callback.answer("❌ Этого мема нет в избранном")

        elif data == "search":
            await callback.answer()
            await callback.message.answer("Введи запрос для поиска мемов 🔍")

        elif data == "help":
            await callback.answer()
            help_text = (
                "🎭 <b>MemeBot - помощь</b>\n\n"
                "<b>Основные команды:</b>\n"
                "• /start - перезапустить бота\n"
                "• /random - случайный мем\n"
                "• /search - поиск мемов\n"
                "• /favorites - мои избранные"
            )
            await callback.message.answer(help_text, reply_markup=create_main_keyboard())

        elif data == "menu":
            await callback.answer()
            await cmd_start(callback.message)

        elif data.startswith("more:"):
            query = data[5:]
            await callback.answer("🔍 Ищу ещё мемы...")
            await perform_search(callback.message.chat.id, query, callback.from_user.id)

        elif data.startswith("fav:"):
            meme_idx = int(data[4:])
            user_favs = set(favorites.get(user_id, []))

            if meme_idx in user_favs:
                await callback.answer("❌ Уже в избранном")
            else:
                user_favs.add(meme_idx)
                favorites[user_id] = list(user_favs)
                save_favorites(favorites)
                await callback.answer("❤️ Добавлено в избранное")

    except Exception as e:
        logger.exception("Ошибка в callback")
        await callback.answer("❌ Ошибка")



async def main():
    logger.info("🚀 Запуск MemeBot...")
    main_menu = setup_main_menu()
    await bot.set_my_commands(main_menu)

    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.exception("Ошибка бота")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())