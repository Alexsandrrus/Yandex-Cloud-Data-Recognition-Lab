import os
import re
import logging
import pandas as pd
from tqdm import tqdm
import pdfplumber
import matplotlib.pyplot as plt
from transformers import pipeline, LayoutLMv2Processor, LayoutLMv2ForTokenClassification
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ROPProcessor:
    def __init__(self):
        """Инициализация моделей с обработкой ошибок"""
        logger.info(f"Используется устройство: {DEVICE}")
        
        # Проверка версий
        logger.info(f"PyTorch версия: {torch.__version__}")
        try:
            from transformers import __version__ as transformers_version
            logger.info(f"Transformers версия: {transformers_version}")
        except ImportError:
            logger.warning("Не удалось определить версию transformers")

        # Инициализация NER модели для русского языка
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="DeepPavlov/rubert-base-cased-ner-ontonotes",
                aggregation_strategy="simple",
                device=DEVICE,
                framework="pt"
            )
            logger.info("Модель для распознавания именованных сущностей (NER) успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки NER модели: {e}")
            raise RuntimeError("Не удалось инициализировать NER модель")

        # Инициализация модели для анализа структуры документов
        try:
            self.processor = LayoutLMv2Processor.from_pretrained(
                "microsoft/layoutlmv2-base-uncased",
                revision="no_ocr"
            )
            self.layout_model = LayoutLMv2ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv2-base-uncased"
            ).to(DEVICE)
            logger.info("Модель для анализа структуры документов успешно загружена")
        except Exception as e:
            logger.warning(f"Ошибка загрузки LayoutLMv2: {e}")
            self.layout_model = None

        self.cache = {}
    
    def extract_text_from_pdf(self, pdf_path):
        """Извлечение текста из PDF с сохранением структуры"""
        if pdf_path in self.cache:
            return self.cache[pdf_path]
            
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    words = page.extract_words(
                        x_tolerance=2,
                        y_tolerance=2,
                        keep_blank_chars=False,
                        use_text_flow=True,
                        extra_attrs=["fontname", "size"]
                    )
                    text.append(words)
            self.cache[pdf_path] = text
        except Exception as e:
            logger.error(f"Ошибка чтения PDF {pdf_path}: {e}")
            raise

        return text
    
    def analyze_layout(self, pdf_path):
        """Анализ структуры документа"""
        if self.layout_model is None:
            return None

        try:
            images = [pdfplumber.open(pdf_path).pages[0].to_image().original]
            encoded_inputs = self.processor(images, return_tensors="pt").to(DEVICE)
            outputs = self.layout_model(**encoded_inputs)
            return outputs
        except Exception as e:
            logger.warning(f"Ошибка анализа структуры {pdf_path}: {e}")
            return None
    
    def extract_goods_info(self, pdf_path):
        """Извлечение информации о товарах из русскоязычных деклараций"""
        text_data = self.extract_text_from_pdf(pdf_path)
        goods_data = []
        
        # Оптимизированные шаблоны для русскоязычных документов
        patterns = {
            'name': r'(?:наименование|описание).*?товар.*?\d+.*?([^\n]+?)(?=\d{10}|код|$)', 
            'hs_code': r'(?:код\s*тн\s*вэд|код\s*товара).*?(\d{6,10})',
            'quantity': r'(?:количество|кол-во).*?(\d+)\s*(?:шт|штук)',
            'net_weight': r'(?:вес\s*нетто|нетто).*?(\d+[,.]?\d*)',
            'gross_weight': r'(?:вес\s*брутто|брутто).*?(\d+[,.]?\d*)'
        }
        
        for page in text_data:
            page_text = " ".join([word['text'] for word in page])
            
            # Нормализация текста для улучшения поиска
            page_text = re.sub(r'\s+', ' ', page_text.lower())
            
            # Поиск товарных позиций
            for match in re.finditer(patterns['name'], page_text, re.IGNORECASE):
                good_block = match.group()
                good_info = {}
                
                try:
                    # Извлечение данных из блока товара
                    good_info['name'] = re.sub(r'\d+\)', '', match.group(1)).strip()
                    
                    # Поиск остальных атрибутов в пределах 200 символов от названия
                    context = page_text[max(0, match.start()-100):match.end()+100]
                    
                    if hs_match := re.search(patterns['hs_code'], context):
                        good_info['hs_code'] = re.sub(r'\D', '', hs_match.group(1))
                    
                    if qty_match := re.search(patterns['quantity'], context):
                        good_info['quantity'] = int(qty_match.group(1).replace(',', ''))
                    
                    if net_match := re.search(patterns['net_weight'], context):
                        good_info['net_weight'] = float(net_match.group(1).replace(',', '.'))
                    
                    if gross_match := re.search(patterns['gross_weight'], context):
                        good_info['gross_weight'] = float(gross_match.group(1).replace(',', '.'))
                    
                    if all(k in good_info for k in ['name', 'hs_code']):
                        goods_data.append(good_info)
                        
                except (AttributeError, ValueError) as e:
                    logger.debug(f"Ошибка обработки блока товара: {e}")
                    continue
        
        return goods_data
    
    def load_rop_reference(self, pdf_path):
        """Загрузка кодов ТН ВЭД из русскоязычного документа"""
        text_data = self.extract_text_from_pdf(pdf_path)
        full_text = " ".join([" ".join([word['text'] for word in page]) for page in text_data])
        
        # Улучшенный поиск кодов ТН ВЭД в русском тексте
        hs_codes = re.findall(r'(?:код\s*тн\s*вэд\s*|тн\s*вэд\s*)?(\d{4}\s?\d{2}\s?\d{2,4})', full_text, re.IGNORECASE)
        
        # Нормализация кодов (удаление пробелов и нецифровых символов)
        return list(set([re.sub(r'\D', '', code) for code in hs_codes))
    
    def process_declaration(self, pdf_path, rop_codes):
        """Обработка одной декларации с улучшенным логированием"""
        try:
            goods = self.extract_goods_info(pdf_path)
            results = []
            
            for good in goods:
                if good.get('hs_code', '') in rop_codes:
                    packaging_weight = good.get('gross_weight', 0) - good.get('net_weight', 0)
                    
                    results.append({
                        'Наименование товара': good.get('name', ''),
                        'Количество': good.get('quantity', 0),
                        'Код ТН ВЭД': good.get('hs_code', ''),
                        'Вес нетто (кг)': good.get('net_weight', 0),
                        'Вес упаковки (кг)': packaging_weight,
                        'Файл': os.path.basename(pdf_path)
                    })
            
            if results:
                logger.info(f"Обработан {pdf_path}: найдено {len(results)} товаров под РОП")
            else:
                logger.debug(f"В {pdf_path} не найдено товаров под РОП")
            
            return results
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_path}: {e}", exc_info=True)
            return []
    
    def process_folder(self, declarations_folder, rop_reference_path, output_file):
        """Основная функция обработки с прогресс-баром"""
        try:
            # Загружаем справочник РОП
            rop_codes = self.load_rop_reference(rop_reference_path)
            logger.info(f"Загружено {len(rop_codes)} кодов ТН ВЭД под РОП")
            
            # Получаем список PDF файлов
            pdf_files = [
                os.path.join(declarations_folder, f) 
                for f in os.listdir(declarations_folder) 
                if f.lower().endswith('.pdf')
            ]
            logger.info(f"Найдено {len(pdf_files)} файлов для обработки")
            
            # Параллельная обработка
            all_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                process_func = partial(self.process_declaration, rop_codes=rop_codes)
                results = list(tqdm(
                    executor.map(process_func, pdf_files),
                    total=len(pdf_files),
                    desc="Обработка деклараций"
                ))
                
                for result in results:
                    all_results.extend(result)
            
            # Создаем итоговый DataFrame
            df = pd.DataFrame(all_results)
            
            if not df.empty:
                # Группируем по коду ТН ВЭД и наименованию
                report = df.groupby(['Код ТН ВЭД', 'Наименование товара']).agg({
                    'Количество': 'sum',
                    'Вес нетто (кг)': 'sum',
                    'Вес упаковки (кг)': 'sum',
                    'Файл': lambda x: ", ".join(set(x))
                }).reset_index()
                
                # Сохраняем отчет
                report.to_excel(output_file, index=False)
                logger.info(f"Отчет сохранен в {output_file}")
                
                # Визуализация
                self.generate_plots(report)
                
                return report
            else:
                logger.warning("Не найдено товаров, попадающих под РОП")
                return None
                
        except Exception as e:
            logger.error(f"Критическая ошибка обработки: {e}", exc_info=True)
            raise
    
    def generate_plots(self, report):
        """Генерация графиков для отчета"""
        try:
            # Топ-10 товаров по весу упаковки
            plt.figure(figsize=(12, 6))
            top_packaging = report.nlargest(10, 'Вес упаковки (кг)')
            plt.barh(
                top_packaging['Наименование товара'].str[:50] + " (" + top_packaging['Код ТН ВЭД'] + ")",
                top_packaging['Вес упаковки (кг)'],
                color='skyblue'
            )
            plt.title('Топ-10 товаров по весу упаковки (РОП)')
            plt.xlabel('Вес упаковки, кг')
            plt.tight_layout()
            plt.savefig('top_packaging_weight.png')
            plt.close()
            
            # Распределение по кодам ТН ВЭД
            plt.figure(figsize=(10, 6))
            report.groupby('Код ТН ВЭД')['Вес нетто (кг)'].sum().plot.pie(
                autopct='%1.1f%%',
                startangle=90,
                counterclock=False
            )
            plt.title('Распределение веса по кодам ТН ВЭД')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig('hs_code_distribution.png')
            plt.close()
            
            logger.info("Графики сохранены в текущую директорию")
        except Exception as e:
            logger.error(f"Ошибка генерации графиков: {e}")

if __name__ == "__main__":
    try:
        processor = ROPProcessor()
        
        # Пути к файлам (замените на актуальные)
        declarations_folder = './2. scr/ДТ'  # Папка с декларациями
        rop_reference = './1. docs/Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf'   # Файл с кодами ТН ВЭД под РОП
        output_file = './2. scr/output'       # Выходной файл
        
        # Запуск обработки
        result = processor.process_folder(declarations_folder, rop_reference, output_file)
        
        if result is not None:
            print("\nРезультаты обработки:")
            print(result.head())
            print(f"\nПолный отчет сохранен в {output_file}")
    except Exception as e:
        logger.error(f"Ошибка выполнения программы: {e}", exc_info=True)
        raise