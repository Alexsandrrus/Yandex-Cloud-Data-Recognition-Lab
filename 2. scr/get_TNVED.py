import os
import re
import logging
import string
from pathlib import Path
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

class PDFTextExtractor:
    def __init__(self):
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.poppler_path = r'C:\Program Files\poppler-24.08.0\Library\bin'
        self._setup_logging()
        self._setup_environment()

    def _setup_logging(self):
        """Настройка системы логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_extractor.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_environment(self):
        """Проверка и настройка зависимостей"""
        # Проверка Tesseract
        if os.path.exists(self.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self.logger.info(f"Tesseract настроен: {self.tesseract_cmd}")
            
            # Проверка русского языка
            tessdata_dir = r'C:\Program Files\Tesseract-OCR\tessdata'
            rus_traineddata = os.path.join(tessdata_dir, 'rus.traineddata')
            if os.path.exists(rus_traineddata):
                self.logger.info("Русский язык обнаружен в Tesseract")
            else:
                self.logger.error(f"Файл rus.traineddata не найден в: {tessdata_dir}")
                self.logger.info("Скачайте его с: https://github.com/tesseract-ocr/tessdata")
                self.logger.info(f"И поместите в: {tessdata_dir}")
        else:
            self.logger.error(f"Tesseract не найден по пути: {self.tesseract_cmd}")

        # Проверка Poppler
        if os.path.exists(self.poppler_path):
            self.logger.info(f"Poppler настроен: {self.poppler_path}")
        else:
            self.logger.error(f"Poppler не найден по пути: {self.poppler_path}")

    def extract_text(self, pdf_path: str) -> str:
        """Основной метод извлечения текста"""
        pdf_path = str(Path(pdf_path).resolve())
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

        methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
            self._extract_with_pdfminer,
            self._extract_with_ocr
        ]
        
        best_text = ""
        for method in methods:
            try:
                current_text = method(pdf_path)
                if self._is_text_better(best_text, current_text):
                    best_text = current_text
                    self.logger.info(f"Метод {method.__name__} дал улучшение")
            except Exception as e:
                self.logger.warning(f"Ошибка в {method.__name__}: {e}")
        
        return self._clean_text(best_text) if best_text else ""

    def _is_text_better(self, old: str, new: str) -> bool:
        """Сравнивает качество текста"""
        if not new:
            return False
        return len(new) > len(old) * 1.1 or self._has_more_cyrillic(new, old)

    def _has_more_cyrillic(self, new: str, old: str) -> bool:
        """Сравнивает количество кириллицы"""
        cyr_new = len(re.findall('[а-яА-ЯёЁ]', new))
        cyr_old = len(re.findall('[а-яА-ЯёЁ]', old))
        return cyr_new > cyr_old * 1.5

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Извлечение с помощью pdfplumber"""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"PDFPlumber error: {e}")
            return ""

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Извлечение с помощью PyPDF2"""
        text = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"PyPDF2 error: {e}")
            return ""

    def _extract_with_pdfminer(self, pdf_path: str) -> str:
        """Извлечение с помощью PDFMiner"""
        text = []
        try:
            for page_layout in extract_pages(pdf_path):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text += element.get_text() + "\n"
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"PDFMiner error: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Извлечение текста с помощью OCR"""
        if not os.path.exists(self.poppler_path):
            self.logger.warning("Poppler не доступен, пропускаем OCR")
            return ""
            
        try:
            text = []
            images = convert_from_path(
                pdf_path,
                poppler_path=self.poppler_path,
                dpi=400,
                grayscale=True,
                thread_count=4
            )
            
            for img in images:
                img = self._enhance_image(img)
                page_text = pytesseract.image_to_string(
                    img,
                    lang='rus+eng',
                    config='--oem 3 --psm 6'
                )
                if page_text:
                    text.append(page_text)
            
            return "\n".join(text)
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return ""

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Улучшение качества изображения для OCR"""
        try:
            # Конвертация в grayscale
            img = img.convert('L')
            
            # Увеличение контраста
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # Бинаризация
            img = img.point(lambda x: 0 if x < 140 else 255)
            
            # Уменьшение шума
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            return img
        except Exception as e:
            self.logger.warning(f"Image enhancement error: {e}")
            return img

    def _clean_text(self, text: str) -> str:
        """Очистка и форматирование текста"""
        if not text:
            return ""
            
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Восстановление абзацев
        text = re.sub(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', '\n\n', text)
        
        # Сохранение только нужных символов
        allowed = (
            string.digits + 
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' +
            'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' +
            r' .,!?()[]{}<>|/\\:;"«»%-–—+\n№'
        )
        text = ''.join(c for c in text if c in allowed)
        
        # Улучшение форматирования чисел
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        
        return text

def main():
    extractor = PDFTextExtractor()
    
    try:
        # Определение путей
        base_dir = Path(__file__).parent.resolve()
        input_pdf = base_dir / '../1. docs/Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf'
        output_dir = base_dir / '../2. scr/output/'
        output_txt = output_dir / 'extracted_text.txt'
        
        # Создание директории
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Извлечение текста
        text = extractor.extract_text(str(input_pdf))
        
        if text:
            # Сохранение результата
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(text)
            
            extractor.logger.info(f"Текст сохранен в: {output_txt}")
            
            # Статистика
            chars = len(text)
            words = len(re.findall(r'\w+', text))
            nums = len(re.findall(r'\d', text))
            
            extractor.logger.info(f"Статистика: {chars} симв., {words} слов, {nums} цифр")
            
            # Пример текста
            sample = text[:500] + ('...' if len(text) > 500 else '')
            extractor.logger.info(f"Пример текста:\n{sample}")
        else:
            extractor.logger.error("Не удалось извлечь текст")
            
    except Exception as e:
        extractor.logger.error(f"Ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()