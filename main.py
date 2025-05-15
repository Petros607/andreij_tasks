import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.enums.parse_mode import ParseMode
from aiogram.types import Message
from aiogram.utils import executor
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)

def query_llama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return "Ошибка при обращении к LLaMA: " + response.text
    except Exception as e:
        return f"Ошибка соединения с Ollama: {e}"

@dp.message_handler(commands=['start'])
async def handle_start(message: Message):
    welcome_text = (
        "Привет! Я — Telegram-бот, подключённый к локальной языковой модели LLaMA 3 🦙\n\n"
        "Просто напиши мне сообщение, и я постараюсь ответить.\n"
        "В любой момент ты можешь ввести /start, чтобы начать диалог заново."
    )
    await message.answer(welcome_text)

@dp.message_handler()
async def handle_message(message: Message):
    if message.text:
        user_text = message.text.strip()
        if user_text:
            await message.answer("🤖 Думаю...")
            response = query_llama(user_text)
            await message.answer(response)
    else:
        await message.answer("Я понимаю только текстовые сообщения.")

if __name__ == '__main__':
    print("Бот запущен!")
    executor.start_polling(dp, skip_updates=True)
