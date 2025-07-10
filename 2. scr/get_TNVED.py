import os
import re
import logging
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self):
        # Укажите правильные пути для вашей системы
        self.poppler_path = r'C:\Program Files\poppler-23.11.0\Library\bin'
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_text(self, pdf_path):
        """Комбинированное извлечение текста (PDF + OCR)"""
        try:
            # Сначала пробуем извлечь текст как обычный PDF
            text = self._extract_with_pdfplumber(pdf_path)
            if len(text) > 200:  # Достаточно текста
                return text
            
            # Если текста мало, пробуем OCR
            ocr_text = self._extract_with_ocr(pdf_path)
            return ocr_text if len(ocr_text) > len(text) else text
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста: {e}")
            return ""

    def _extract_with_pdfplumber(self, pdf_path):
        """Извлечение текста из PDF напрямую"""
        import pdfplumber
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception as e:
            logger.warning(f"PDFPlumber error: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path):
        """Извлечение текста через OCR"""
        try:
            images = convert_from_path(
                pdf_path,
                poppler_path=self.poppler_path,
                dpi=400,
                thread_count=4
            )
            full_text = []
            for img in images:
                processed = self._preprocess_image(img)
                text = pytesseract.image_to_string(
                    processed,
                    lang='rus+eng',
                    config='--psm 6 --oem 3'
                )
                full_text.append(text)
            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""

    def _preprocess_image(self, image):
        """Улучшение качества изображения для OCR"""
        try:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Улучшение контраста
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Бинаризация
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Удаление шума
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return Image.fromarray(cleaned)
        except Exception as e:
            logger.warning(f"Image processing error: {e}")
            return image

class RegulationParser:
    def __init__(self):
        self.extractor = PDFTextExtractor()
        # Улучшенные шаблоны для поиска данных
        self.pattern = re.compile(
            r'(?P<name>.*?[^\d])\n'  # Наименование товара
            r'.*?(?:упаковка|тар[аы]).*?\n'  # Упаковка
            r'.*?ТН\s*ВЭД\s*ЕАЭС\s*(?P<code>\d{4}[\.\s]?\d{2}[\.\s]?\d{2,4})',  # Код
            re.IGNORECASE | re.DOTALL
        )

    def parse_regulation(self, pdf_path):
        """Анализ постановления и извлечение данных"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        text = self.extractor.extract_text(pdf_path)
        
        # Сохраняем извлеченный текст для отладки
        os.makedirs("debug", exist_ok=True)
        with open("debug/extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Извлекаем данные
        items = []
        for match in self.pattern.finditer(text):
            name = re.sub(r'\s+', ' ', match.group('name')).strip()
            code = re.sub(r'[^\d]', '', match.group('code'))
            code = code.ljust(10, '0')[:10]
            
            items.append({
                'Наименование товара, упаковки': name,
                'Код единой Товарной номенклатуры внешнеэкономической деятельности Евразийского экономического союза(ТН ВЭД ЕАЭС)': code
            })
        
        return items

    def save_to_excel(self, items, output_path):
        """Сохранение результатов в Excel"""
        if not items:
            logger.warning("Нет данных для сохранения")
            return False
            
        df = pd.DataFrame(items)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Создаем Excel файл с настройками
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Товары')
            
            # Настройка ширины колонок
            worksheet = writer.sheets['Товары']
            worksheet.set_column('A:A', 60)  # Широкая колонка для наименования
            worksheet.set_column('B:B', 15)  # Узкая колонка для кода
        
        logger.info(f"Результаты сохранены в: {output_path}")
        return True

def main():
    try:
        parser = RegulationParser()
        
        # Конфигурация путей
        config = {
            'input': '../1. docs/Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf',
            'output': '../2. scr/output/Результаты_анализа.xlsx'
        }
        
        # Нормализация путей
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for key in config:
            config[key] = os.path.normpath(os.path.join(base_dir, config[key]))
        
        # Проверка путей
        if not os.path.exists(config['input']):
            raise FileNotFoundError(f"Файл постановления не найден: {config['input']}")
        
        # Анализ документа
        logger.info("Анализ постановления...")
        items = parser.parse_regulation(config['input'])
        
        if items:
            logger.info(f"Найдено записей: {len(items)}")
            
            # Сохранение результатов
            parser.save_to_excel(items, config['output'])
            
            # Вывод примеров
            print("\nПримеры найденных записей:")
            for item in items[:3]:
                print(f"{item['Наименование товара, упаковки'][:50]}... | {item['Код единой Товарной номенклатуры внешнеэкономической деятельности Евразийского экономического союза(ТН ВЭД ЕАЭС)']}")
        else:
            logger.warning("Не найдено данных в документе")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()