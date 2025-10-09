"""
Утилиты для извлечения текста из файлов
"""

import docx
import PyPDF2
import asyncio

async def extract_txt(file_path: str) -> str:
    """Извлечение текста из TXT файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp1251') as file:
            return file.read()

async def extract_docx(file_path: str) -> str:
    """Извлечение текста из DOCX файла"""
    try:
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        raise Exception(f"Ошибка чтения DOCX: {str(e)}")

async def extract_pdf(file_path: str) -> str:
    """Извлечение текста из PDF файла"""
    try:
        text = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    except Exception as e:
        raise Exception(f"Ошибка чтения PDF: {str(e)}")