
import os
import re
import logging
import csv
from pathlib import Path
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from fpdf import FPDF

class PDFDataExtractor:
    def __init__(self):
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.poppler_path = r'C:\Program Files\poppler-24.08.0\Library\bin'
        # Путь к папке со шрифтами
        self.font_dir = Path(__file__).parent / "djsans"
        self._setup_logging()
        self._setup_environment()

        # Проверка существования папки со шрифтами
        if not self.font_dir.exists():
            self.logger.warning(f"Папка со шрифтами не найдена: {self.font_dir}")

    def _setup_logging(self):
        """Настройка системы логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_data_extractor.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_environment(self):
        """Проверка и настройка зависимостей"""
        if os.path.exists(self.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self.logger.info(f"Tesseract настроен: {self.tesseract_cmd}")
            
            tessdata_dir = r'C:\Program Files\Tesseract-OCR\tessdata'
            rus_traineddata = os.path.join(tessdata_dir, 'rus.traineddata')
            if not os.path.exists(rus_traineddata):
                self.logger.warning(f"Файл rus.traineddata не найден в: {tessdata_dir}")

        if not os.path.exists(self.poppler_path):
            self.logger.warning(f"Poppler не найден по пути: {self.poppler_path}")

    def extract_data_from_folder(self, input_folder: str, output_folder: str):
        """Извлечение данных из всех PDF в папке и сохранение в CSV и PDF"""
        input_path = Path(input_folder).resolve()
        output_path = Path(output_folder).resolve()
        
        if not input_path.exists():
            raise FileNotFoundError(f"Папка с PDF не найдена: {input_path}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / 'extracted_data.csv'
        
        # Подготовка CSV файла
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Файл',
                'Маркировка и количество',
                'Код товара (ТН ВЭД)',
                'Вес брутто (кг)',
                'Вес нетто (кг)'
            ])
            
            # Создаем список для данных PDF
            data_rows = []
            
            # Обработка каждого PDF файла
            for pdf_file in input_path.glob('*.pdf'):
                try:
                    self.logger.info(f"Обработка файла: {pdf_file.name}")
                    text = self.extract_text(str(pdf_file))
                    
                    if text:
                        data = self._parse_data(text, pdf_file.name)
                        row_data = [
                            pdf_file.name,
                            data.get('marking', 'Не найдено'),
                            data.get('code', 'Не найдено'),
                            data.get('gross_weight', 'Не найдено'),
                            data.get('net_weight', 'Не найдено')
                        ]
                        
                        writer.writerow(row_data)
                        data_rows.append(row_data)
                        self.logger.info(f"Данные извлечены из {pdf_file.name}")
                    else:
                        self.logger.warning(f"Не удалось извлечь текст из: {pdf_file.name}")
                        writer.writerow([pdf_file.name, 'Ошибка чтения', '', '', ''])
                        data_rows.append([pdf_file.name, 'Ошибка чтения', '', '', ''])
                        
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке {pdf_file.name}: {str(e)}")
                    writer.writerow([pdf_file.name, 'Ошибка обработки', '', '', ''])
                    data_rows.append([pdf_file.name, 'Ошибка обработки', '', '', ''])
        
        # Сохраняем в PDF
        self._save_to_pdf(data_rows, output_file)
        self.logger.info("Обработка завершена. Данные сохранены в CSV и PDF файлы.")

    def _save_to_pdf(self, data_rows, output_csv_path):
        """Сохраняет данные в PDF файл"""
        pdf_path = output_csv_path.with_suffix('.pdf')
        
        try:
            # Создаем PDF документ
            pdf = FPDF()
            pdf.add_page()
            
            # Пути к шрифтам
            dejavu_path = self.font_dir / "DejaVuSansCondensed.ttf"
            arial_path = self.font_dir / "arial.ttf"
            
            # Пробуем загрузить DejaVu
            try:
                if dejavu_path.exists():
                    pdf.add_font('DejaVu', '', str(dejavu_path), uni=True)
                    pdf.set_font('DejaVu', '', 10)
                    self.logger.info("Используется шрифт DejaVu")
                else:
                    raise FileNotFoundError(f"DejaVu не найден по пути: {dejavu_path}")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить DejaVu: {str(e)}. Пробую Arial.")
                if arial_path.exists():
                    pdf.add_font('Arial', '', str(arial_path), uni=True)
                    pdf.set_font('Arial', '', 10)
                    self.logger.info("Используется шрифт Arial")
                else:
                    self.logger.warning("Arial также не найден. Использую стандартный шрифт (возможны проблемы с кириллицей).")
                    pdf.set_font("Arial", size=10)  # Стандартный шрифт PDF (не поддерживает кириллицу)
            
            # Заголовок отчета
            pdf.set_font_size(14)
            pdf.cell(0, 10, "Отчет по извлеченным данным из PDF", 0, 1, 'C')
            pdf.ln(10)
            pdf.set_font_size(10)
            
            # Заголовки столбцов
            headers = ['Файл', 'Маркировка и количество', 'Код товара', 'Вес брутто', 'Вес нетто']
            col_widths = [30, 70, 25, 25, 25]
            
            # Добавляем заголовки
            pdf.set_fill_color(200, 220, 255)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, fill=True)
            pdf.ln()
            
            # Добавляем данные
            pdf.set_fill_color(255, 255, 255)
            for row in data_rows:
                for i, value in enumerate(row):
                    # Укорачиваем длинный текст маркировки
                    if i == 1 and len(value) > 150:
                        value = value[:150] + "..."
                    pdf.cell(col_widths[i], 10, value, border=1)
                pdf.ln()
            
            # Сохраняем PDF
            pdf.output(str(pdf_path))
            self.logger.info(f"PDF отчет сохранен: {pdf_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при создании PDF: {str(e)}")

    def _save_to_pdf(self, data_rows, output_csv_path):
        """Сохраняет данные в PDF файл"""
        pdf_path = output_csv_path.with_suffix('.pdf')
        
        try:
            # Создаем PDF документ
            pdf = FPDF()
            pdf.add_page()
            
            # Добавляем шрифт с поддержкой кириллицы
            try:
                pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                pdf.set_font('DejaVu', '', 10)
            except:
                pdf.add_font('Arial', '', 'arial.ttf', uni=True)
                pdf.set_font('Arial', '', 10)
            
            # Заголовок отчета
            pdf.set_font_size(14)
            pdf.cell(0, 10, "Отчет по извлеченным данным из PDF", 0, 1, 'C')
            pdf.ln(10)
            pdf.set_font_size(10)
            
            # Заголовки столбцов
            headers = ['Файл', 'Маркировка и количество', 'Код товара', 'Вес брутто', 'Вес нетто']
            col_widths = [30, 70, 25, 25, 25]
            
            # Добавляем заголовки
            pdf.set_fill_color(200, 220, 255)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, fill=True)
            pdf.ln()
            
            # Добавляем данные
            pdf.set_fill_color(255, 255, 255)
            for row in data_rows:
                for i, value in enumerate(row):
                    # Укорачиваем длинный текст маркировки
                    if i == 1 and len(value) > 150:
                        value = value[:150] + "..."
                    pdf.cell(col_widths[i], 10, value, border=1)
                pdf.ln()
            
            # Сохраняем PDF
            pdf.output(str(pdf_path))
            self.logger.info(f"PDF отчет сохранен: {pdf_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при создании PDF: {str(e)}")

    def extract_text(self, pdf_path: str) -> str:
        """Извлечение текста из PDF с приоритетом для OCR"""
        methods = [
            self._extract_with_ocr,
            self._extract_with_pdfplumber,
            self._extract_with_pdfminer,
            self._extract_with_pypdf2,
        ]
        
        best_text = ""
        for method in methods:
            try:
                current_text = method(pdf_path)
                if self._is_text_better(best_text, current_text):
                    best_text = current_text
                    self.logger.info(f"Метод {method.__name__} дал улучшение")
            except Exception as e:
                self.logger.warning(f"Ошибка в {method.__name__}: {str(e)}")
        
        return best_text

    def _parse_data(self, text: str, filename: str) -> dict:
        """Улучшенный парсинг текста для извлечения нужных данных"""
        data = {
            'marking': 'Не найдено',
            'code': 'Не найдено',
            'gross_weight': 'Не найдено',
            'net_weight': 'Не найдено'
        }
        
        # Сохраняем оригинальный текст для отладки
        debug_file = Path('debug') / f'{filename}.txt'
        debug_file.parent.mkdir(exist_ok=True)
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        try:
            # 1. Маркировка и количество - УЛУЧШЕННЫЙ ПОИСК
            marking_patterns = [
                # Основной шаблон: захватываем содержимое после заголовка
                r'(?:31\s*[^\n]*(?:Грузовые места|Описание товаров|Маркировка и количество)|'
                r'Маркировка и количество[^\n]*|Описание товаров[^\n]*)\s*'
                r'[:—\-\s]*\s*(.*?)\s*(?=\d{1,2}\s*Товар|\d{1,2}\s*Код|Вес|33\s|35\s|38\s|$)',
                
                # Шаблон для формата с нумерацией 1)
                r'1[\)\.\-]\s*(.*?)\s*(?=\d{1,2}\s*Товар|\d{1,2}\s*Код|Вес|33\s|35\s|38\s|$)',
                
                # Альтернативные варианты начала описания
                r'(?:Наименование товара|Описание):\s*(.*?)\s*(?=\d{1,2}\s*Товар|\d{1,2}\s*Код|Вес|$)',
                
                # Поиск по характерным фразам (более гибкий)
                r'(?:Маркировка и количество[^\n]*|Описание товаров[^\n]*)\s*[\r\n]+(.*?)\s*(?=\d{1,2}\s*[^\n]*Код|Вес|$)',
                
                # Резервный шаблон: ищем любой текст после заголовка до цифровых разделов
                r'(?:31|33|35|38)[^\n]*\s*[\r\n]+(.*?)\s*(?=\d{1,2}\s*[^\n]*(?:Товар|Код|Вес)|$)',
                
                # Новый шаблон для сложных случаев с финансовой информацией
                r'11\d{1,2}\s*[^\n]*\s*(.*?)\s*(?=\d{1,2}\s*[^\n]*(?:Товар|Код|Вес|Счет|Вид|Место))'
            ]
            
            # Расширенный поиск с обработкой многострочных описаний
            for pattern in marking_patterns:
                marking_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if marking_match:
                    # Выбираем последнюю группу, где находится нужный текст
                    groups = marking_match.groups()
                    marking_text = groups[-1].strip() if groups else ""
                    
                    # Удаляем технические коды вида (796)
                    marking_text = re.sub(r'\(\d+\)', '', marking_text)
                    
                    # Очистка текста - улучшенная обработка переносов
                    marking_text = re.sub(r'-\s*\n', '', marking_text)  # Соединяем перенесенные слова
                    marking_text = re.sub(r'\s+', ' ', marking_text)     # Убираем лишние пробелы
                    marking_text = re.sub(r'(\d)\s+(\d)', r'\1\2', marking_text)  # Объединяем разделенные цифры
                    
                    # Удаляем возможные артефакты OCR
                    marking_text = re.sub(r'[^\w\s\d.,;:()\-\/\n]', ' ', marking_text)
                    
                    # Удаляем оставшиеся фрагменты заголовков
                    header_phrases = [
                        'Грузовые места', 'Описание товаров', 
                        'Маркировка и количество', 'Номера контейнеров',
                        'Количество и отличительные особенности'
                    ]
                    for phrase in header_phrases:
                        marking_text = re.sub(rf'\b{re.escape(phrase)}\b', '', marking_text, flags=re.IGNORECASE)
                    
                    marking_text = re.sub(r'\s+', ' ', marking_text).strip()
                    
                    if marking_text and len(marking_text) > 10:  # Проверка на минимальную значимую длину
                        data['marking'] = marking_text
                        self.logger.info(f"Найдена маркировка по шаблону: {pattern[:50]}...")
                        break
            
            # 2. Код товара (ТН ВЭД)
            code_patterns = [
                r'33\s*Код\s*товара[\s:—\-]*(\d{10})',
                r'ТН\s*ВЭД[\s:—\-]*(\d{10})',
                r'Код\s*ТН\s*ВЭД[\s:—\-]*(\d{10})',
                r'\b(\d{10})\b(?!\.\d)'
            ]
            
            for pattern in code_patterns:
                code_match = re.search(pattern, text)
                if code_match:
                    data['code'] = code_match.group(1)
                    break
            
            # 3. Вес брутто (кг)
            gross_patterns = [
                r'35\s*Вес\s*брутто\s*\(кг\)[^0-9]*([\d\s,\.]+)',
                r'Вес\s*брутто\s*\(кг\)[^0-9]*([\d\s,\.]+)',
                r'Брутто\s*\(кг\)[^0-9]*([\d\s,\.]+)'
            ]
            
            for pattern in gross_patterns:
                gross_match = re.search(pattern, text, re.IGNORECASE)
                if gross_match:
                    gross_value = gross_match.group(1).strip()
                    gross_value = re.sub(r'\s', '', gross_value).replace(',', '.')
                    if re.match(r'\d+\.?\d*', gross_value):
                        data['gross_weight'] = gross_value
                        break
            
            # 4. Вес нетто (кг)
            net_patterns = [
                r'38\s*Вес\s*нетто\s*\(кг\)[^0-9]*([\d\s,\.]+)',
                r'Вес\s*нетто\s*\(кг\)[^0-9]*([\d\s,\.]+)',
                r'Нетто\s*\(кг\)[^0-9]*([\d\s,\.]+)'
            ]
            
            for pattern in net_patterns:
                net_match = re.search(pattern, text, re.IGNORECASE)
                if net_match:
                    net_value = net_match.group(1).strip()
                    net_value = re.sub(r'\s', '', net_value).replace(',', '.')
                    if re.match(r'\d+\.?\d*', net_value):
                        data['net_weight'] = net_value
                        break
            
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге данных: {str(e)}")
        
        self.logger.info(f"Извлеченные данные для {filename}: {data}")
        return data
        
    def _is_text_better(self, old: str, new: str) -> bool:
        """Сравнивает качество текста"""
        if not new:
            return False
            
        # Считаем количество значимых символов
        def count_significant_chars(txt):
            cyr = len(re.findall('[а-яА-ЯёЁ]', txt))
            num = len(re.findall('\d', txt))
            return cyr + num
            
        old_score = count_significant_chars(old)
        new_score = count_significant_chars(new)
        
        return new_score > old_score * 1.2

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Извлечение с помощью pdfplumber с обработкой таблиц"""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Пробуем извлечь текст из таблиц
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text.append(" ".join([cell.strip() for cell in row if cell]))
                    
                    # Извлекаем обычный текст
                    page_text = page.extract_text() or ""
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"PDFPlumber error: {str(e)}")
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
            self.logger.warning(f"PyPDF2 error: {str(e)}")
            return ""

    def _extract_with_pdfminer(self, pdf_path: str) -> str:
        """Извлечение с помощью PDFMiner с улучшенной обработкой"""
        text = []
        try:
            for page_layout in extract_pages(pdf_path):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        element_text = element.get_text().strip()
                        if element_text:
                            element_text = re.sub(r'([а-яА-ЯёЁ])([А-ЯЁ])', r'\1 \2', element_text)
                            page_text += element_text + "\n"
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"PDFMiner error: {str(e)}")
            return ""

    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Извлечение текста с помощью OCR с улучшенной обработкой изображений"""
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
                thread_count=2
            )
            
            for i, img in enumerate(images):
                img = self._enhance_image(img)
                debug_img = Path('debug') / f'{Path(pdf_path).stem}_{i}.png'
                debug_img.parent.mkdir(exist_ok=True)
                img.save(debug_img)
                
                page_text = pytesseract.image_to_string(
                    img,
                    lang='rus+eng',
                    config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
                )
                if page_text:
                    text.append(page_text)
            
            return "\n".join(text)
        except Exception as e:
            self.logger.error(f"OCR error: {str(e)}")
            return ""

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Улучшение качества изображения для OCR"""
        try:
            # Увеличение размера
            img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.LANCZOS)
            
            # Конвертация в grayscale
            img = img.convert('L')
            
            # Увеличение контраста
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # Резкость
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            # Бинаризация
            img = img.point(lambda x: 0 if x < 160 else 255)
            
            # Уменьшение шума
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            return img
        except Exception as e:
            self.logger.warning(f"Image enhancement error: {str(e)}")
            return img

def main():
    extractor = PDFDataExtractor()
    
    try:
        base_dir = Path(__file__).parent.resolve()
        input_folder = base_dir / 'ДТ'  # Папка с PDF-файлами
        output_folder = base_dir / 'output'  # Папка для результатов
        
        # Создаем папки если их нет
        input_folder.mkdir(exist_ok=True)
        output_folder.mkdir(exist_ok=True)
        (base_dir / 'debug').mkdir(exist_ok=True)
        
        extractor.extract_data_from_folder(str(input_folder), str(output_folder))
        
    except Exception as e:
        extractor.logger.error(f"Ошибка: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()