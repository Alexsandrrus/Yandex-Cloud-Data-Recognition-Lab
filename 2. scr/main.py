import os
import re
import logging
import pandas as pd
from tqdm import tqdm
import pdfplumber
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ROPProcessor:
    def __init__(self):
        """Упрощенная инициализация без сложных моделей"""
        logger.info("Инициализация процессора РОП")
        self.cache = {}
    
    def extract_text_from_pdf(self, pdf_path):
        """Извлечение текста из PDF с базовой обработкой ошибок"""
        if pdf_path in self.cache:
            return self.cache[pdf_path]
            
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Пробуем разные методы извлечения текста
                    page_text = page.extract_text()
                    if not page_text:
                        page_text = " ".join([word['text'] for word in page.extract_words()])
                    text.append(page_text)
            self.cache[pdf_path] = text
        except Exception as e:
            logger.error(f"Ошибка чтения PDF {pdf_path}: {e}")
            raise

        return text
    
    def _extract_value(self, text, pattern):
        """Улучшенное извлечение числовых значений"""
        match = re.search(pattern, text)
        if match:
            try:
                # Обрабатываем разные форматы чисел
                num_str = match.group(1).replace(',', '.').replace(' ', '')
                return float(num_str)
            except (ValueError, AttributeError):
                return 0.0
        return 0.0
    
    def _extract_good_name(self, text, hs_code):
        """Улучшенное извлечение наименования товара"""
        # Ищем текст между "Описание товаров" и кодом ТН ВЭД
        patterns = [
            r'Описание\s*товаров?\s*(.*?)\d{6,10}',
            r'Наименование\s*товара\s*(.*?)\d{6,10}',
            r'31\s*(.*?)\d{6,10}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Очистка от лишних символов
                name = re.sub(r'[\n\t\r]+', ' ', name)
                name = re.sub(r'\s{2,}', ' ', name)
                return name
        return f"Товар {hs_code}"
    
    def extract_goods_info(self, pdf_path):
        """Улучшенное извлечение информации о товарах"""
        text_data = self.extract_text_from_pdf(pdf_path)
        goods_data = []
        
        for page_text in text_data:
            # Нормализация текста
            page_text = re.sub(r'\s+', ' ', page_text)
            
            # Улучшенные шаблоны для поиска кодов товаров
            hs_code_patterns = [
                r'Код\s*товара\s*(\d{6,10})',
                r'ТН\s*ВЭД\s*(\d{6,10})',
                r'33\s*(\d{6,10})'
            ]
            
            for pattern in hs_code_patterns:
                for hs_match in re.finditer(pattern, page_text):
                    hs_code = hs_match.group(1)
                    context_start = max(0, hs_match.start()-500)
                    context_end = hs_match.end()+500
                    context = page_text[context_start:context_end]
                    
                    good_info = {
                        'hs_code': hs_code,
                        'name': self._extract_good_name(context, hs_code),
                        'quantity': self._extract_value(context, r'(Количество|41)\s*(\d+[,.]?\d*)'),
                        'net_weight': self._extract_value(context, r'(Вес\s*нетто|38)\s*(\d+[,.]?\d*)'),
                        'gross_weight': self._extract_value(context, r'(Вес\s*брутто|35)\s*(\d+[,.]?\d*)')
                    }
                    
                    if len(hs_code) >= 6:  # Минимальная длина кода
                        goods_data.append(good_info)
        
        return goods_data
    
    def load_rop_reference(self, pdf_path):
        """Улучшенная загрузка кодов ТН ВЭД из постановления"""
        if not os.path.exists(pdf_path):
            logger.error(f"Файл с кодами не найден: {pdf_path}")
            return []
        
        try:
            full_text = "\n".join(self.extract_text_from_pdf(pdf_path))
            
            # Улучшенные шаблоны для поиска кодов
            code_patterns = [
                r'(?:ТН\s*ВЭД\s*ЕАЭС|код\s*ТН\s*ВЭД)\s*(\d{4}\s?\d{2}\s?\d{2,4})',
                r'\b\d{4}[\.\s]?\d{2}[\.\s]?\d{2,4}\b',
                r'Приложение\s*[№N]\s*\d+\s*.*?(\d{6,10})'
            ]
            
            found_codes = set()
            for pattern in code_patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    code = re.sub(r'[^\d]', '', match.group(1) if match.groups() else match.group())
                    if 6 <= len(code) <= 10:
                        found_codes.add(code[:10].ljust(10, '0'))  # Нормализация до 10 цифр
            
            logger.info(f"Найдено {len(found_codes)} кодов ТН ВЭД ЕАЭС")
            return list(found_codes)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кодов: {e}")
            return []
    
    def process_declaration(self, pdf_path, rop_codes):
        """Обработка одной декларации с улучшенным логированием"""
        try:
            goods = self.extract_goods_info(pdf_path)
            results = []
            
            for good in goods:
                # Проверяем полное совпадение или начало кода
                if any(good['hs_code'].startswith(code) for code in rop_codes):
                    results.append({
                        'Наименование товара': good['name'],
                        'Код ТН ВЭД': good['hs_code'],
                        'Количество (шт)': good['quantity'],
                        'Вес нетто (кг)': good['net_weight'],
                        'Вес упаковки (кг)': max(0, good['gross_weight'] - good['net_weight']),
                        'Файл': os.path.basename(pdf_path)
                    })
            
            if results:
                logger.info(f"Обработан {pdf_path}: найдено {len(results)} товаров под РОП")
            else:
                logger.debug(f"В {pdf_path} не найдено товаров под РОП")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_path}: {e}")
            return []
    
    def process_folder(self, declarations_folder, rop_reference_path, output_file):
        """Основной метод обработки с улучшенной обработкой ошибок"""
        try:
            # Загрузка кодов РОП
            rop_codes = self.load_rop_reference(rop_reference_path)
            if not rop_codes:
                logger.error("Не загружены коды ТН ВЭД ЕАЭС. Проверьте файл с постановлением.")
                return None
            
            logger.info(f"Загружены коды РОП: {rop_codes[:5]}... (всего {len(rop_codes)})")
            
            # Получение списка PDF файлов
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
                process_func = partial(self.process_declaration, rop_codes=rop_codes)
                results = list(tqdm(
                    executor.map(process_func, pdf_files),
                    total=len(pdf_files),
                    desc="Обработка деклараций"
                ))
                for result in results:
                    all_results.extend(result)
            
            # Формирование отчета
            if all_results:
                df = pd.DataFrame(all_results)
                
                # Группировка с учетом возможных None значений
                report = df.groupby(['Код ТН ВЭД', 'Наименование товара']).agg({
                    'Количество (шт)': 'sum',
                    'Вес нетто (кг)': 'sum',
                    'Вес упаковки (кг)': 'sum',
                    'Файл': lambda x: ", ".join(set(x.dropna()))
                }).reset_index()
                
                # Сохранение отчета
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                report.to_excel(output_file, index=False)
                logger.info(f"Отчет сохранен в {output_file}")
                
                # Дополнительная информация
                total_weight = report['Вес нетто (кг)'].sum()
                logger.info(f"Всего найдено {len(report)} позиций, общий вес: {total_weight:.2f} кг")
                
                return report
            else:
                logger.warning("Не найдено товаров под РОП. Проверьте коды в постановлении.")
                return None
                
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            raise

if __name__ == "__main__":
    try:
        processor = ROPProcessor()
        
        # Укажите правильные пути к файлам
        declarations_folder = './2. scr/Выгрузки по импорту'
        rop_reference = './1. docs/Постановление Правительства РФ о РОП.pdf'
        output_file = './2. scr/output/ROP_report.xlsx'
        
        # Проверка путей
        if not os.path.exists(declarations_folder):
            raise FileNotFoundError(f"Папка с декларациями не найдена: {declarations_folder}")
        if not os.path.exists(rop_reference):
            raise FileNotFoundError(f"Файл с кодами РОП не найден: {rop_reference}")
        
        # Запуск обработки
        result = processor.process_folder(declarations_folder, rop_reference, output_file)
        
        if result is not None:
            print("\nУспешно завершено! Результаты:")
            print(result.head())
            print(f"\nПолный отчет сохранен в {output_file}")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения программы: {e}")