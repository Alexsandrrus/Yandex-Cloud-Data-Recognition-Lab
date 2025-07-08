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

        # Инициализация NER модели (опционально)
        self.ner_pipeline = None
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="Davlan/bert-base-multilingual-cased-ner-hrl",
                aggregation_strategy="simple",
                device=DEVICE,
                framework="pt"
            )
            logger.info("NER модель успешно загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить NER модель: {e}")

        # Инициализация модели для анализа структуры документов
        self.layout_model = None
        try:
            self.processor = LayoutLMv2Processor.from_pretrained(
                "microsoft/layoutlmv2-base-uncased",
                revision="no_ocr"
            )
            self.layout_model = LayoutLMv2ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv2-base-uncased"
            ).to(DEVICE)
            logger.info("Модель для анализа структуры документов загружена")
        except Exception as e:
            logger.warning(f"Ошибка загрузки LayoutLMv2: {e}")

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
    
    def _extract_value(self, text, pattern):
        """Извлечение числового значения из текста"""
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                return 0.0
        return 0.0
    
    def _extract_good_name(self, text, hs_code):
        """Извлечение наименования товара"""
        # Ищем текст между "Описание товаров" и кодом ТН ВЭД
        match = re.search(r'31\s*Описание товаров\s*(.*?)\d{10}', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return f"Товар {hs_code}"
    
    def extract_goods_info(self, pdf_path):
        """Извлечение информации о товарах из деклараций"""
        text_data = self.extract_text_from_pdf(pdf_path)
        goods_data = []
        
        for page in text_data:
            page_text = " ".join([word['text'] for word in page])
            
            # Ищем все коды товаров на странице
            hs_code_matches = re.finditer(r'33\s*Код товара\s*(\d{10})', page_text)
            
            for hs_match in hs_code_matches:
                hs_code = hs_match.group(1)
                context = page_text[max(0, hs_match.start()-300):hs_match.end()+300]
                
                good_info = {
                    'hs_code': hs_code,
                    'name': self._extract_good_name(context, hs_code),
                    'net_weight': self._extract_value(context, r'38\s*Вес нетто\s*\(кг\)\s*(\d+[,.]?\d*)'),
                    'gross_weight': self._extract_value(context, r'35\s*Вес брутто\s*\(кг\)\s*(\d+[,.]?\d*)')
                }
                
                if good_info['hs_code']:  # Если найден код товара
                    goods_data.append(good_info)
        
        return goods_data
    
    def load_rop_reference(self, pdf_path):
        """Загрузка кодов ТН ВЭД ЕАЭС из постановления"""
        if not os.path.exists(pdf_path):
            logger.error(f"Файл с кодами не найден: {pdf_path}")
            return []
        
        try:
            full_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
            
            # Ищем коды в разных форматах
            code_patterns = [
                r'(?:ТН\s*ВЭД\s*ЕАЭС|код\s*ТН\s*ВЭД\s*ЕАЭС)\s*(\d{6,10})',
                r'\b\d{4}\s?\d{2}\s?\d{2,4}\b'
            ]
            
            found_codes = set()
            for pattern in code_patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    code = re.sub(r'\D', '', match.group(1) if match.groups() else match.group())
                    if len(code) >= 6:  # Минимум 6 цифр
                        found_codes.add(code.zfill(10))  # Нормализуем до 10 цифр
            
            logger.info(f"Найдено {len(found_codes)} кодов ТН ВЭД ЕАЭС")
            return list(found_codes)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кодов: {e}")
            return []
    
    def process_declaration(self, pdf_path, rop_codes):
        """Обработка одной декларации"""
        try:
            goods = self.extract_goods_info(pdf_path)
            results = []
            
            for good in goods:
                if good['hs_code'] in rop_codes:
                    results.append({
                        'Наименование товара': good['name'],
                        'Код ТН ВЭД': good['hs_code'],
                        'Вес нетто (кг)': good['net_weight'],
                        'Вес упаковки (кг)': good['gross_weight'] - good['net_weight'],
                        'Файл': os.path.basename(pdf_path)
                    })
            
            if results:
                logger.info(f"Обработан {pdf_path}: найдено {len(results)} товаров под РОП")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_path}: {e}")
            return []
    
    def process_folder(self, declarations_folder, rop_reference_path, output_file):
        """Обработка всех деклараций в папке"""
        try:
            # Загрузка кодов РОП
            rop_codes = self.load_rop_reference(rop_reference_path)
            if not rop_codes:
                logger.error("Не загружены коды ТН ВЭД ЕАЭС")
                return None
            
            # Получение списка PDF файлов
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
            
            # Формирование отчета
            if all_results:
                df = pd.DataFrame(all_results)
                report = df.groupby(['Код ТН ВЭД', 'Наименование товара']).agg({
                    'Вес нетто (кг)': 'sum',
                    'Вес упаковки (кг)': 'sum',
                    'Файл': lambda x: ", ".join(set(x))
                }).reset_index()
                
                report.to_excel(output_file, index=False)
                logger.info(f"Отчет сохранен в {output_file}")
                
                # Генерация графиков
                self._generate_report_plots(report)
                return report
            else:
                logger.warning("Не найдено товаров под РОП")
                return None
                
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            raise
    
    def _generate_report_plots(self, report):
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
            plt.title('Распределение веса по кодам ТН ВЭД ЕАЭС')
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
        
        # Пути к файлам
        declarations_folder = './2. scr/ДТ'
        rop_reference = './1. docs/Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf'
        output_file = './2. scr/output.xlsx'
        
        # Проверка существования файлов
        if not os.path.exists(declarations_folder):
            raise FileNotFoundError(f"Папка с декларациями не найдена: {declarations_folder}")
        if not os.path.exists(rop_reference):
            raise FileNotFoundError(f"Файл с кодами РОП не найден: {rop_reference}")
        
        # Запуск обработки
        result = processor.process_folder(declarations_folder, rop_reference, output_file)
        
        if result is not None:
            print("\nРезультаты обработки:")
            print(result.head())
            print(f"\nПолный отчет сохранен в {output_file}")
            print("Графики сохранены в текущей директории")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения программы: {e}")