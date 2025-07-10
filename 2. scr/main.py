import os
import re
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pdfplumber
import cv2
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Укажите путь к tesseract если он не в PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PDFTextExtractor:
    def __init__(self):
        self.cache = {}
        
    def extract_text(self, pdf_path):
        """Комбинированный метод извлечения текста (PDF + OCR)"""
        if pdf_path in self.cache:
            return self.cache[pdf_path]
            
        try:
            # Сначала пробуем извлечь текст напрямую
            text = self._extract_with_pdfplumber(pdf_path)
            
            # Если текста мало, пробуем OCR
            if len(text) < 100:
                ocr_text = self._extract_with_ocr(pdf_path)
                if len(ocr_text) > len(text):
                    text = ocr_text
                    
            self.cache[pdf_path] = text
            return text
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {pdf_path}: {e}")
            return []

    def _extract_with_pdfplumber(self, pdf_path):
        """Извлечение текста из PDF с помощью pdfplumber"""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    page_text = " ".join([word['text'] for word in page.extract_words()])
                text.append(page_text)
        return text

    def _extract_with_ocr(self, pdf_path):
        """Извлечение текста с помощью OCR (для сканированных PDF)"""
        text = []
        try:
            images = self._convert_pdf_to_images(pdf_path)
            for img in images:
                # Предварительная обработка изображения
                processed = self._preprocess_image(img)
                # Извлечение текста с помощью Tesseract
                page_text = pytesseract.image_to_string(processed, lang='rus+eng')
                text.append(page_text)
        except Exception as e:
            logger.error(f"OCR error: {e}")
        return text

    def _convert_pdf_to_images(self, pdf_path, dpi=300):
        """Конвертация PDF в изображения"""
        try:
            from pdf2image import convert_from_path
            return convert_from_path(pdf_path, dpi=dpi)
        except ImportError:
            logger.error("Для OCR требуется pdf2image: pip install pdf2image")
            return []
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF в изображения: {e}")
            return []

    def _preprocess_image(self, image):
        """Улучшение изображения для OCR"""
        try:
            # Конвертация в OpenCV формат
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Увеличение контраста
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Бинаризация
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Удаление шума
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return Image.fromarray(cleaned)
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            return image

class ROPProcessor:
    def __init__(self):
        logger.info("Инициализация процессора РОП")
        self.extractor = PDFTextExtractor()
        self.cache = {}

    def load_rop_reference(self, pdf_path):
        """Загрузка кодов ТН ВЭД с улучшенным поиском"""
        if not os.path.exists(pdf_path):
            logger.error(f"Файл с кодами не найден: {pdf_path}")
            return []
        
        try:
            text_pages = self.extractor.extract_text(pdf_path)
            full_text = "\n".join(text_pages)
            
            # Сохраним текст для отладки
            debug_file = os.path.join("debug", os.path.basename(pdf_path) + ".txt")
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # Улучшенные шаблоны для табличных данных
            table_patterns = [
                r'(?:код\s*ТН\s*ВЭД|ТН\s*ВЭД\s*ЕАЭС)\s*(\d{6,10})',
                r'\b\d{4}\.?\d{2}\.?\d{2,4}\b',
                r'(?<!\d)(\d{4})\s?(\d{2})\s?(\d{2,4})(?!\d)'
            ]
            
            found_codes = set()
            for pattern in table_patterns:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    code = "".join([g for g in match.groups() if g]) if match.groups() else match.group()
                    code = re.sub(r'[^\d]', '', code)
                    if 6 <= len(code) <= 10:
                        normalized = code.ljust(10, '0')[:10]
                        found_codes.add(normalized)
            
            if not found_codes:
                # Альтернативный метод поиска в таблицах
                found_codes.update(self._find_codes_in_tables(pdf_path))
            
            logger.info(f"Найдено {len(found_codes)} кодов ТН ВЭД")
            return sorted(found_codes)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кодов: {e}")
            return []

    def _find_codes_in_tables(self, pdf_path):
        """Поиск кодов в таблицах PDF"""
        codes = set()
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            for cell in row:
                                if cell and isinstance(cell, str):
                                    matches = re.findall(r'\b\d{6,10}\b', cell)
                                    codes.update(matches)
        except Exception as e:
            logger.error(f"Ошибка извлечения таблиц: {e}")
        return codes

    def extract_goods_info(self, pdf_path):
        """Извлечение информации о товарах с улучшенной обработкой"""
        text_pages = self.extractor.extract_text(pdf_path)
        goods_data = []
        
        for page_text in text_pages:
            # Нормализация текста
            page_text = re.sub(r'\s+', ' ', page_text)
            
            # Поиск кодов товаров
            hs_codes = re.findall(r'\b\d{6,10}\b', page_text)
            for hs_code in hs_codes:
                # Поиск контекста вокруг кода
                context = self._get_code_context(page_text, hs_code)
                
                good_info = {
                    'hs_code': hs_code,
                    'name': self._extract_good_name(context, hs_code),
                    'quantity': self._extract_value(context, r'(Количество|41)\s*[:=]?\s*(\d+[\.,]?\d*)'),
                    'net_weight': self._extract_value(context, r'(Вес\s*нетто|38)\s*[:=]?\s*(\d+[\.,]?\d*)'),
                    'gross_weight': self._extract_value(context, r'(Вес\s*брутто|35)\s*[:=]?\s*(\d+[\.,]?\d*)')
                }
                
                if len(hs_code) >= 6:
                    goods_data.append(good_info)
        
        return goods_data

    def _get_code_context(self, text, code, window=500):
        """Получение контекста вокруг кода"""
        pos = text.find(code)
        start = max(0, pos - window)
        end = min(len(text), pos + len(code) + window)
        return text[start:end]

    def _extract_good_name(self, text, hs_code):
        """Извлечение наименования товара"""
        patterns = [
            r'Наименование\s*товара[^\w]*([^\d]+?)\d{6,10}',
            r'Описание\s*товара[^\w]*([^\d]+?)\d{6,10}',
            r'31[^\w]*([^\d]+?)\d{6,10}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\n\t\r]+', ' ', name)
                name = re.sub(r'\s{2,}', ' ', name)
                return name
        return f"Товар {hs_code}"

    def _extract_value(self, text, pattern):
        """Извлечение числовых значений"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(2) if match.groups() > 1 else match.group(1)
                num_str = num_str.replace(',', '.').replace(' ', '')
                return float(num_str)
            except (ValueError, AttributeError):
                return 0.0
        return 0.0

    def process_folder(self, declarations_folder, rop_reference_path, output_file):
        """Обработка папки с декларациями"""
        try:
            # Загрузка кодов РОП
            rop_codes = self.load_rop_reference(rop_reference_path)
            if not rop_codes:
                logger.error("Не удалось загрузить коды ТН ВЭД")
                return None
            
            logger.info(f"Загружены коды РОП. Примеры: {rop_codes[:5]}")
            
            # Поиск PDF файлов
            pdf_files = []
            for root, _, files in os.walk(declarations_folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            if not pdf_files:
                logger.error("Не найдено PDF файлов для обработки")
                return None
                
            logger.info(f"Найдено {len(pdf_files)} файлов для обработки")
            
            # Обработка файлов
            all_results = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                process_func = partial(self._process_declaration, rop_codes=rop_codes)
                results = list(tqdm(
                    executor.map(process_func, pdf_files),
                    total=len(pdf_files),
                    desc="Обработка деклараций"
                ))
                all_results = [item for sublist in results for item in sublist]
            
            # Формирование отчета
            if all_results:
                df = pd.DataFrame(all_results)
                report = df.groupby(['Код ТН ВЭД', 'Наименование товара']).agg({
                    'Количество (шт)': 'sum',
                    'Вес нетто (кг)': 'sum',
                    'Вес упаковки (кг)': 'sum',
                    'Файл': lambda x: ", ".join(set(x.dropna()))
                }).reset_index()
                
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                report.to_excel(output_file, index=False)
                logger.info(f"Отчет сохранен в {output_file}")
                return report
            else:
                logger.warning("Не найдено товаров под РОП")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            raise

    def _process_declaration(self, pdf_path, rop_codes):
        """Обработка одной декларации"""
        try:
            goods = self.extract_goods_info(pdf_path)
            results = []
            
            for good in goods:
                if any(good['hs_code'].startswith(code) for code in rop_codes):
                    results.append({
                        'Наименование товара': good['name'],
                        'Код ТН ВЭД': good['hs_code'],
                        'Количество (шт)': good['quantity'],
                        'Вес нетто (кг)': good['net_weight'],
                        'Вес упаковки (кг)': max(0, good['gross_weight'] - good['net_weight']),
                        'Файл': os.path.basename(pdf_path)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_path}: {e}")
            return []

if __name__ == "__main__":
    try:
        processor = ROPProcessor()
        
        # Пути к файлам (используйте raw строки или двойные слеши)
        declarations_folder = os.path.normpath(r'2. scr/ДТ')
        rop_reference = os.path.normpath(r'1. docs/Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf')
        output_file = os.path.normpath(r'2. scr/output/ROP_report.xlsx')
        
        # Проверка путей
        if not os.path.exists(declarations_folder):
            raise FileNotFoundError(f"Папка не найдена: {declarations_folder}")
        if not os.path.exists(rop_reference):
            raise FileNotFoundError(f"Файл не найден: {rop_reference}")
        
        # Создаем папку для отладки
        os.makedirs("debug", exist_ok=True)
        
        # Запуск обработки
        result = processor.process_folder(declarations_folder, rop_reference, output_file)
        
        if result is not None:
            print("\nУспешно завершено! Результаты:")
            print(result.head())
            print(f"\nПолный отчет сохранен в {output_file}")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")