"""
Telegram бот для проверки лабораторных работ через 3 бесплатные AI модели
"""

import asyncio
import os
import tempfile
from pathlib import Path
import time

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from dotenv import load_dotenv
from openai import OpenAI

from file_utils import extract_docx, extract_pdf, extract_txt


# Загрузка переменных окружения
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    print("Добавьте TELEGRAM_BOT_TOKEN в .env файл!")
    exit(1)

# Инициализация
bot = Bot(
    token=BOT_TOKEN, 
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# Настройки бесплатных моделей
FREE_MODELS = {
    "Qwen 2.5 72B": {
        "provider": "openrouter",
        "model": "qwen/qwen2.5-72b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", "free")
    },
    "Llama 3.3 70B": {
        "provider": "openrouter", 
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", "free")
    },
    "DeepSeek Coder": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-coder:free",
        "base_url": "https://openrouter.ai/api/v1", 
        "api_key": os.getenv("OPENROUTER_API_KEY", "free")
    }
}

# Поддерживаемые форматы
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt']
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ

# Глобальная переменная для хранения извлеченного контента
user_files = {}

def get_main_keyboard():
    """Главная клавиатура"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❓ Помощь", callback_data="help"),
         InlineKeyboardButton(text="🤖 Модели", callback_data="models")],
        [InlineKeyboardButton(text="⚡ Тест всех моделей", callback_data="test_all")]
    ])

def get_models_keyboard():
    """Клавиатура выбора модели"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Qwen 2.5 72B", callback_data="model_qwen")],
        [InlineKeyboardButton(text="🦙 Llama 3.3 70B", callback_data="model_llama")],
        [InlineKeyboardButton(text="💻 DeepSeek Coder", callback_data="model_deepseek")],
        [InlineKeyboardButton(text="⚡ Все модели", callback_data="model_all")],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_main")]
    ])

@dp.message(CommandStart())
async def start_command(message: Message):
    """Команда /start"""
    await message.answer(
        f"🎓 <b>Привет, {message.from_user.first_name}!</b>\n\n"
        "Я проверяю лабораторные работы с помощью 3 бесплатных AI моделей!\n\n"
        "📄 <b>Как пользоваться:</b>\n"
        "1. Отправь файл с работой\n"
        "2. Выбери модель для проверки\n"
        "3. Получи детальный анализ\n\n"
        "📁 <b>Поддерживаю:</b> PDF, DOCX, TXT файлы (до 20 МБ)\n"
        "🤖 <b>Доступные модели:</b> Qwen 2.5 72B, Llama 3.3 70B, DeepSeek Coder\n\n"
        "<i>Можешь протестировать все модели сразу для сравнения! ⚡</i>",
        reply_markup=get_main_keyboard()
    )

@dp.message(Command("help"))
async def help_command(message: Message):
    """Команда /help"""
    await message.answer(
        "📖 <b>Как пользоваться ботом:</b>\n\n"
        "1️⃣ Отправь мне файл с лабораторной работой\n"
        "2️⃣ Выбери AI модель для проверки\n"
        "3️⃣ Жди результат (1-3 минуты)\n"
        "4️⃣ Получи детальную оценку\n\n"
        "<b>📁 Поддерживаемые форматы:</b>\n"
        "• PDF (до 20 МБ)\n"
        "• DOCX (Microsoft Word)\n"
        "• TXT (текстовые файлы)\n\n"
        "<b>🤖 Доступные модели:</b>\n"
        "• <b>Qwen 2.5 72B</b> - универсальная модель\n"
        "• <b>Llama 3.3 70B</b> - от Meta, хороша для текста\n"  
        "• <b>DeepSeek Coder</b> - специализирована на код\n\n"
        "<b>📊 Что проверяется:</b>\n"
        "• Качество кода и решения\n"
        "• Полнота документации\n"
        "• Правильность выводов\n"
        "• Оформление работы\n\n"
        "<b>💡 Результат:</b> Оценка из 100 баллов + рекомендации",
        reply_markup=get_main_keyboard()
    )

@dp.message(Command("models"))
async def models_command(message: Message):
    """Команда /models - информация о моделях"""
    await show_models_info(message)

async def show_models_info(message: Message):
    """Показать информацию о моделях"""
    models_text = "🤖 <b>Доступные AI модели:</b>\n\n"
    
    for model_name, config in FREE_MODELS.items():
        models_text += f"<b>{model_name}</b>\n"
        if model_name == "Qwen 2.5 72B":
            models_text += "• Универсальная модель от Alibaba\n• Хороша для общего анализа\n• 72 миллиарда параметров\n\n"
        elif model_name == "Llama 3.3 70B":
            models_text += "• Модель от Meta (Facebook)\n• Отличные текстовые способности\n• 70 миллиардов параметров\n\n"
        elif model_name == "DeepSeek Coder":
            models_text += "• Специализирована на программирование\n• Лучше всего для анализа кода\n• Понимает多种 языков программирования\n\n"
    
    models_text += "⚡ <b>Совет:</b> Используй 'Тест всех моделей' для сравнения!"
    
    await message.answer(
        models_text,
        reply_markup=get_models_keyboard()
    )

@dp.callback_query(F.data == "help")
async def help_callback(callback: CallbackQuery):
    """Помощь через callback"""
    await help_command(callback.message)
    await callback.answer()

@dp.callback_query(F.data == "models")
async def models_callback(callback: CallbackQuery):
    """Модели через callback"""
    await show_models_info(callback.message)
    await callback.answer()

@dp.callback_query(F.data == "test_all")
async def test_all_callback(callback: CallbackQuery):
    """Тест всех моделей через callback"""
    await callback.message.answer(
        "⚡ <b>Режим тестирования всех моделей</b>\n\n"
        "Отправь файл с лабораторной работой, и я запущу проверку "
        "во всех трех моделях одновременно!\n\n"
        "Это займет 2-3 минуты, но ты получишь сравнение результатов.",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "back_main")
async def back_main_callback(callback: CallbackQuery):
    """Возврат в главное меню"""
    await callback.message.answer(
        "🔙 <b>Главное меню</b>",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()

@dp.message(F.text & ~F.text.startswith('/'))
async def handle_text(message: Message):
    """Обработка текстовых сообщений"""
    await message.answer(
        "📄 <b>Отправь файл для проверки!</b>\n\n"
        "Я не анализирую текстовые сообщения.\n"
        "Просто прикрепи файл с лабораторной работой.\n\n"
        "Поддерживаю: PDF, DOCX, TXT\n\n"
        "<i>После загрузки файла выбери модель для анализа</i>",
        reply_markup=get_main_keyboard()
    )

@dp.message(F.document)
async def handle_document(message: Message):
    """Обработка файла"""
    document = message.document
    file_name = document.file_name
    file_size = document.file_size
    
    # Проверка размера
    if file_size > MAX_FILE_SIZE:
        await message.answer(
            f"❌ <b>Файл слишком большой!</b>\n\n"
            f"📏 Размер файла: {file_size / 1024 / 1024:.1f} МБ\n"
            f"📏 Максимум: {MAX_FILE_SIZE / 1024 / 1024} МБ\n\n"
            f"Попробуй сжать файл или выбрать другой.",
            reply_markup=get_main_keyboard()
        )
        return
    
    # Проверка формата
    file_ext = Path(file_name).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        await message.answer(
            f"❌ <b>Неподдерживаемый формат!</b>\n\n"
            f"📄 Твой файл: <code>{file_ext}</code>\n"
            f"📁 Поддерживаю: <code>{', '.join(SUPPORTED_FORMATS)}</code>\n\n"
            f"Преобразуй файл в подходящий формат.",
            reply_markup=get_main_keyboard()
        )
        return
    
    # Показываем статус обработки
    status_msg = await message.answer(
        "⏳ <b>Обрабатываю файл...</b>\n\n"
        "🔄 Загружаю файл\n"
        "⏳ Извлекаю содержимое\n\n"
        "<i>После этого выбери модель для анализа</i>"
    )
    
    temp_path = None
    user_id = message.from_user.id
    
    try:
        # Загружаем файл
        file = await bot.get_file(document.file_id)
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            await bot.download_file(file.file_path, tmp_file.name)
            temp_path = tmp_file.name
        
        await status_msg.edit_text(
            "⏳ <b>Обрабатываю файл...</b>\n\n"
            "✅ Файл загружен\n"
            "🔄 Извлекаю содержимое\n"
            "⏳ Подготовка к анализу"
        )
        
        # Извлекаем содержимое в зависимости от типа файла
        if file_ext == '.txt':
            content = await extract_txt(temp_path)
        elif file_ext == '.docx':
            content = await extract_docx(temp_path)
        elif file_ext == '.pdf':
            content = await extract_pdf(temp_path)
        else:
            raise Exception("Неподдерживаемый формат")
        
        # Проверяем, что содержимое не пустое
        if not content.strip():
            raise Exception("Файл пуст или не содержит читаемого текста")
        
        # Сохраняем контент для пользователя
        user_files[user_id] = {
            'content': content,
            'file_name': file_name,
            'file_size': file_size,
            'status_msg': status_msg
        }
        
        # Удаляем временный файл
        if temp_path:
            os.unlink(temp_path)
        
        await status_msg.edit_text(
            f"✅ <b>Файл готов к анализу!</b>\n\n"
            f"📄 <b>Файл:</b> <code>{file_name}</code>\n"
            f"📊 <b>Размер:</b> {file_size / 1024:.1f} КБ\n"
            f"📝 <b>Символов:</b> {len(content):,}\n\n"
            f"<b>Выбери AI модель для проверки:</b>",
            reply_markup=get_models_keyboard()
        )
        
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        await status_msg.edit_text(
            f"❌ <b>Ошибка при обработке файла:</b>\n\n"
            f"<code>{str(e)}</code>\n\n"
            f"Попробуй еще раз или выбери другой файл.",
            reply_markup=get_main_keyboard()
        )

async def check_with_ai(content: str, model_name: str) -> str:
    """Проверка работы через выбранную AI модель"""
    
    model_config = FREE_MODELS[model_name]
    
    prompt = f"""
Проанализируй эту лабораторную работу студента и дай развернутую оценку.

КРИТЕРИИ ОЦЕНКИ (100 баллов максимум):
1. Качество кода и решения (0-30 баллов)
2. Полнота и правильность реализации (0-30 баллов)  
3. Документация и комментарии (0-20 баллов)
4. Оформление и структура работы (0-20 баллов)

СОДЕРЖИМОЕ РАБОТЫ:
{content[:12000]}  # Ограничиваем длину для бесплатных моделей

Дай конструктивную оценку:
- Кратко опиши, что делает работа
- Оцени каждый критерий с обоснованием
- Укажи сильные стороны
- Дай конкретные рекомендации для улучшения
- Поставь итоговую оценку из 100 баллов

Будь справедлив, но требователен. Пиши понятно для студтеля.
"""
    
    try:
        client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"] or "free"
        )
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        
        processing_time = time.time() - start_time
        
        result = response.choices[0].message.content
        return f"{result}\n\n⏱️ <i>Время обработки: {processing_time:.1f} сек.</i>"
        
    except Exception as e:
        return f"❌ Ошибка при обращении к {model_name}: {str(e)}\n\nПопробуйте еще раз через несколько минут."

async def check_with_all_models(content: str, file_info: dict) -> list:
    """Проверка работы через все модели одновременно"""
    tasks = []
    
    for model_name in FREE_MODELS.keys():
        task = asyncio.create_task(check_with_ai(content, model_name))
        tasks.append((model_name, task))
    
    results = []
    for model_name, task in tasks:
        try:
            result = await task
            results.append((model_name, result))
        except Exception as e:
            results.append((model_name, f"❌ Ошибка: {str(e)}"))
    
    return results

@dp.callback_query(F.data.startswith("model_"))
async def process_model_selection(callback: CallbackQuery):
    """Обработка выбора модели и запуск проверки"""
    user_id = callback.from_user.id
    
    if user_id not in user_files:
        await callback.message.answer(
            "❌ <b>Файл не найден!</b>\n\n"
            "Пожалуйста, сначала отправь файл с лабораторной работой.",
            reply_markup=get_main_keyboard()
        )
        await callback.answer()
        return
    
    file_data = user_files[user_id]
    content = file_data['content']
    file_name = file_data['file_name']
    
    model_type = callback.data.replace("model_", "")
    
    # Удаляем сообщение с выбором модели
    await callback.message.delete()
    
    if model_type == "all":
        # Запускаем проверку всеми моделями
        status_msg = await callback.message.answer(
            "⚡ <b>Запускаю проверку всеми моделями...</b>\n\n"
            "🔄 Qwen 2.5 72B - запущена\n"
            "🔄 Llama 3.3 70B - запущена\n" 
            "🔄 DeepSeek Coder - запущена\n\n"
            "<i>Это займет 2-3 минуты...</i>"
        )
        
        results = await check_with_all_models(content, file_data)
        
        # Отправляем результаты
        await status_msg.edit_text(
            f"✅ <b>Проверка завершена!</b>\n\n"
            f"📄 <b>Файл:</b> <code>{file_name}</code>\n"
            f"📊 <b>Символов:</b> {len(content):,}\n"
            f"🤖 <b>Протестировано моделей:</b> {len(results)}"
        )
        
        # Отправляем результаты каждой модели
        for model_name, result in results:
            # Добавляем заголовок модели к результату
            formatted_result = f"🤖 <b>{model_name}:</b>\n\n{result}"
            
            if len(formatted_result) > 4000:
                parts = [formatted_result[i:i+4000] for i in range(0, len(formatted_result), 4000)]
                for i, part in enumerate(parts):
                    if i == 0:
                        await callback.message.answer(part)
                    else:
                        await callback.message.answer(f"📋 <b>Продолжение {model_name} ({i+1}):</b>\n\n{part}")
            else:
                await callback.message.answer(formatted_result)
        
    else:
        # Определяем выбранную модель
        model_map = {
            "qwen": "Qwen 2.5 72B",
            "llama": "Llama 3.3 70B", 
            "deepseek": "DeepSeek Coder"
        }
        
        selected_model = model_map.get(model_type)
        if not selected_model:
            await callback.answer()
            return
        
        # Запускаем проверку одной моделью
        status_msg = await callback.message.answer(
            f"⏳ <b>Проверяю работу...</b>\n\n"
            f"🤖 Модель: {selected_model}\n"
            f"🔄 Анализирую содержимое\n\n"
            f"<i>Обычно занимает 1-2 минуты</i>"
        )
        
        result = await check_with_ai(content, selected_model)
        
        await status_msg.edit_text(
            f"✅ <b>Проверка завершена!</b>\n\n"
            f"📄 <b>Файл:</b> <code>{file_name}</code>\n"
            f"🤖 <b>Модель:</b> {selected_model}\n"
            f"📝 <b>Символов:</b> {len(content):,}"
        )
        
        # Отправляем результат
        formatted_result = f"📋 <b>Результат проверки ({selected_model}):</b>\n\n{result}"
        
        if len(formatted_result) > 4000:
            parts = [formatted_result[i:i+4000] for i in range(0, len(formatted_result), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    await callback.message.answer(part)
                else:
                    await callback.message.answer(f"📋 <b>Продолжение ({i+1}):</b>\n\n{part}")
        else:
            await callback.message.answer(formatted_result)
    
    # Предлагаем проверить еще
    await callback.message.answer(
        "🎉 <b>Готово!</b>\n\n"
        "Можешь отправить еще один файл для проверки или протестировать другие модели 📄",
        reply_markup=get_main_keyboard()
    )
    
    # Очищаем сохраненные данные
    if user_id in user_files:
        del user_files[user_id]
    
    await callback.answer()

# Запуск бота
async def main():
    print("🤖 Запускаю бот для проверки лабораторных работ...")
    print("📄 Поддерживаемые форматы: PDF, DOCX, TXT")
    print("🤖 Доступные модели: Qwen 2.5 72B, Llama 3.3 70B, DeepSeek Coder")
    print("⚡ Режим: Тестирование 3 бесплатных моделей")
    print("-" * 50)
    
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        print("\n Бот остановлен пользователем")
    except Exception as e:
        print(f"\n Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())