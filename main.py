from aiogram.filters.command import Command
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram import F
from funcs.nlp import detect_breed
from funcs.cv import classify_breed
from funcs.loader import load_data
import os


load_data()

TOKEN = os.getenv('TOKEN')
bot = Bot(token=TOKEN)
dp = Dispatcher()


@dp.message(Command(commands=['start']))
async def process_command_start(message: Message):
    detailed_explanation = ("\n\nПривет! Я бот Пушистые Друзья!:\n"
                            "Скинь '1' - И я смогу по фотографии определить породу твоего питомца и подсказать, какой за ним нужен уход!\n"
                            "Скинь '2' - А потом опиши мне питомца, которого ты бы хотел завести и я смогу помочь с выбором породы.")
    await message.answer(detailed_explanation)


@dp.message(F.text.in_({'1', '2'}))
async def choose_mode(message: Message):
    if message.text == '1':
        await message.answer("Режим выбран! Отправь мне фотоку своего любимца.")
    elif message.text == '2':
        await message.answer("Режим выбран! Опиши пушистого друга, которого ты хочешь завести.")


@dp.message(F.content_type == 'photo')
async def handle_photo(message: Message):
    photo = message.photo[-1]  
    photo_file = await bot.get_file(photo.file_id)
    photo_data = await bot.download_file(photo_file.file_path)
    breed = classify_breed(photo_data) 
    await message.answer(breed)
    await choose_mode(message) 


@dp.message(F.text & ~F.text.in_({'1', '2'})) 
async def handle_text(message: Message):
    nlp_result = detect_breed(message.text)  
    await message.answer(nlp_result)
    await choose_mode(message) 


if __name__ == '__main__':
    dp.run_polling(bot)