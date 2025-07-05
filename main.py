import os
import re
import pandas as pd
from tqdm import tqdm
import pdfplumber
import matplotlib.pyplot as plt
from transformers import pipeline, LayoutLMv2Processor, LayoutLMv2ForTokenClassification
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Настройки
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ROPProcessor:
    def __init__(self):
        # Инициализация моделей
        self.ner_pipeline = pipeline(
            "ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
            device=DEVICE
        )
        
        # Модель для анализа структуры документов
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.layout_model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased").to(DEVICE)
        
        # Кэш для хранения промежуточных результатов
        self.cache = {}
        
    def extract_text_from_pdf(self, pdf_path):
        """Улучшенное извлечение текста с сохранением структуры"""
        if pdf_path in self.cache:
            return self.cache[pdf_path]
            
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Извлечение текста с координатами
                words = page.extract_words(
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=["fontname", "size"]
                )
                text.append(words)
        
        self.cache[pdf_path] = text
        return text
    
    def analyze_layout(self, pdf_path):
        """Анализ структуры документа с помощью LayoutLMv2"""
        try:
            images = [pdfplumber.open(pdf_path).pages[0].to_image().original]
            encoded_inputs = self.processor(images, return_tensors="pt").to(DEVICE)
            outputs = self.layout_model(**encoded_inputs)
            return outputs
        except Exception as e:
            print(f"Ошибка анализа структуры: {e}")
            return None
    
    def extract_goods_info(self, pdf_path):
        """Извлечение информации о товарах с улучшенной обработкой"""
        text_data = self.extract_text_from_pdf(pdf_path)
        goods_data = []
        
        # Шаблоны для поиска данных в русскоязычных декларациях
        patterns = {
            'name': r'31\s*Грузовые места и описание товаров[\s\S]*?32Товар\s*\d+[\s\S]*?33Код товара',
            'hs_code': r'33Код товара\s*(\d{10})',
            'quantity': r'Кол-во\s*(\d+)\s*ШТ',
            'net_weight': r'38Вес нетто \(кг\)[\s\S]*?(\d+\.?\d*)',
            'gross_weight': r'35Вес брутто \(кг\)[\s\S]*?(\d+\.?\d*)'
        }
        
        # Обработка каждой страницы
        for page in text_data:
            page_text = " ".join([word['text'] for word in page])
            
            # Поиск товарных позиций
            goods_matches = re.finditer(patterns['name'], page_text)
            for match in goods_matches:
                good_block = match.group()
                good_info = {}
                
                # Извлечение данных из блока товара
                try:
                    good_info['name'] = re.search(r'1\)(.*?)(?=\d+\)|$)', good_block).group(1).strip()
                    good_info['hs_code'] = re.search(patterns['hs_code'], good_block).group(1)
                    good_info['quantity'] = int(re.search(patterns['quantity'], good_block).group(1))
                    good_info['net_weight'] = float(re.search(patterns['net_weight'], good_block).group(1))
                    good_info['gross_weight'] = float(re.search(patterns['gross_weight'], good_block).group(1))
                    
                    goods_data.append(good_info)
                except (AttributeError, ValueError) as e:
                    continue
        
        return goods_data
    
    def load_rop_reference(self, pdf_path):
        """Загрузка кодов ТН ВЭД из постановления с улучшенной обработкой"""
        text_data = self.extract_text_from_pdf(pdf_path)
        full_text = " ".join([" ".join([word['text'] for word in page]) for page in text_data])
        
        # Улучшенный поиск кодов ТН ВЭД в русском тексте
        hs_codes = re.findall(r'(?:код\s*ТН\s*ВЭД\s*|ТН\s*ВЭД\s*)?(\d{4}\s?\d{2}\s?\d{2})', full_text, re.IGNORECASE)
        
        # Нормализация кодов (удаление пробелов и нецифровых символов)
        return [re.sub(r'\D', '', code) for code in hs_codes]
    
    def process_declaration(self, pdf_path, rop_codes):
        """Обработка одной декларации"""
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
            
            return results
        except Exception as e:
            print(f"Ошибка обработки {pdf_path}: {e}")
            return []
    
    def process_folder(self, declarations_folder, rop_reference_path, output_file):
        """Основная функция обработки"""
        # Загружаем справочник РОП
        rop_codes = self.load_rop_reference(rop_reference_path)
        print(f"Загружено {len(rop_codes)} кодов ТН ВЭД под РОП")
        
        # Получаем список PDF файлов
        pdf_files = [
            os.path.join(declarations_folder, f) 
            for f in os.listdir(declarations_folder) 
            if f.lower().endswith('.pdf')
        ]
        
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
            
            # Визуализация
            self.generate_plots(report)
            
            return report
        else:
            print("Не найдено товаров, попадающих под РОП")
            return None
    
    def generate_plots(self, report):
        """Генерация графиков для отчета"""
        try:
            # Топ-10 товаров по весу упаковки
            plt.figure(figsize=(12, 6))
            top_packaging = report.nlargest(10, 'Вес упаковки (кг)')
            plt.barh(
                top_packaging['Наименование товара'].str[:50] + " (" + top_packaging['Код ТН ВЭД'] + ")",
                top_packaging['Вес упаковки (кг)']
            )
            plt.title('Топ-10 товаров по весу упаковки (РОП)')
            plt.xlabel('Вес упаковки, кг')
            plt.tight_layout()
            plt.savefig('top_packaging_weight.png')
            plt.close()
            
            # Распределение по кодам ТН ВЭД
            plt.figure(figsize=(10, 6))
            report.groupby('Код ТН ВЭД')['Вес нетто (кг)'].sum().plot.pie(autopct='%1.1f%%')
            plt.title('Распределение веса по кодам ТН ВЭД')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig('hs_code_distribution.png')
            plt.close()
            
        except Exception as e:
            print(f"Ошибка генерации графиков: {e}")

# Пример использования
if __name__ == "__main__":
    processor = ROPProcessor()
    
    # Пути к файлам (замените на актуальные)
    declarations_folder = './ДТ'
    rop_reference = './Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf'
    output_file = './output'
    
    # Запуск обработки
    result = processor.process_folder(declarations_folder, rop_reference, output_file)
    
    if result is not None:
        print(f"Отчет сохранен в {output_file}")
        print(f"Всего обработано {len(result)} позиций под РОП")