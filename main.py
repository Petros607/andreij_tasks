import logging
import requests
import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram import Router

load_dotenv(encoding="utf-8")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

print(TELEGRAM_TOKEN)
print(OLLAMA_API_URL)

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

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

@router.message(F.text == "/start")
async def handle_start(message: Message):
    welcome_text = (
        "Привет! Я — Telegram-бот, подключённый к локальной языковой модели LLaMA 3 🦙\n\n"
        "Просто напиши мне сообщение, и я постараюсь ответить.\n"
        "В любой момент ты можешь ввести /start, чтобы начать диалог заново."
    )
    await message.answer(welcome_text)

@router.message(F.text)
async def handle_message(message: Message):
    if message.text:
        user_text = message.text.strip()
        if user_text:
            await message.answer("🤖 Думаю...")
            response = query_llama(user_text)
            await message.answer(response)
    else:
        await message.answer("Я понимаю только текстовые сообщения.")

async def main():
    print("Бот запущен!")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
