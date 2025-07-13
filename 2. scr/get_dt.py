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

class PDFDataExtractor:
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
        """Извлечение данных из всех PDF в папке и сохранение в CSV"""
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
            
            # Обработка каждого PDF файла
            for pdf_file in input_path.glob('*.pdf'):
                try:
                    self.logger.info(f"Обработка файла: {pdf_file.name}")
                    text = self.extract_text(str(pdf_file))
                    
                    if text:
                        data = self._parse_data(text)
                        writer.writerow([
                            pdf_file.name,
                            data.get('marking', 'Не найдено'),
                            data.get('code', 'Не найдено'),
                            data.get('gross_weight', 'Не найдено'),
                            data.get('net_weight', 'Не найдено')
                        ])
                        self.logger.info(f"Данные извлечены из {pdf_file.name}")
                    else:
                        self.logger.warning(f"Не удалось извлечь текст из: {pdf_file.name}")
                        writer.writerow([pdf_file.name, 'Ошибка чтения', '', '', ''])
                        
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке {pdf_file.name}: {str(e)}")
                    writer.writerow([pdf_file.name, 'Ошибка обработки', '', '', ''])

    def extract_text(self, pdf_path: str) -> str:
        """Извлечение текста из PDF"""
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
            except Exception as e:
                self.logger.warning(f"Ошибка в {method.__name__}: {str(e)}")
        
        return best_text

    def _parse_data(self, text: str) -> dict:
        """Парсинг текста для извлечения нужных данных"""
        self.logger.debug(f"Анализируемый текст (первые 500 символов):\n{text[:500]}...")
        data = {}
        
        # Маркировка и количество
        marking_match = re.search(
            r'Маркировка\s*и\s*количество\s*[—\-:]\s*(.*?)(?:\n|$)', 
            text, re.IGNORECASE
        )
        data['marking'] = marking_match.group(1).strip() if marking_match else 'Не найдено'
        
        # Код товара (ТН ВЭД)
        code_match = re.search(
            r'(?:33\s*Код\s*товара|ТН\s*ВЭД|Код\s*ТН\s*ВЭД)\s*[—\-:]\s*(\d{10})', 
            text, re.IGNORECASE
        )
        if not code_match:
            code_match = re.search(r'\b\d{10}\b', text)
        
        data['code'] = (code_match.group(1) if code_match and code_match.lastindex else (
                       code_match.group(0) if code_match else 'Не найдено'))
        
        # Вес брутто
        gross_match = re.search(
            r'(?:35\s*Вес\s*брутто|Вес\s*брутто)\s*\(кг\)\s*[—\-:]\s*([\d,\.]+)', 
            text, re.IGNORECASE
        )
        data['gross_weight'] = gross_match.group(1).replace(',', '.') if gross_match else 'Не найдено'
        
        # Вес нетто
        net_match = re.search(
            r'(?:38\s*Вес\s*нетто|Вес\s*нетто)\s*\(кг\)\s*[—\-:]\s*([\d,\.]+)', 
            text, re.IGNORECASE
        )
        data['net_weight'] = net_match.group(1).replace(',', '.') if net_match else 'Не найдено'
        
        self.logger.debug(f"Извлеченные данные: {data}")
        return data

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
            self.logger.warning(f"PDFMiner error: {str(e)}")
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
                dpi=300,
                grayscale=True,
                thread_count=2
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
            self.logger.error(f"OCR error: {str(e)}")
            return ""

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Улучшение качества изображения для OCR"""
        try:
            img = img.convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.point(lambda x: 0 if x < 180 else 255)
            img = img.filter(ImageFilter.MedianFilter(size=1))
            return img
        except Exception as e:
            self.logger.warning(f"Image enhancement error: {str(e)}")
            return img

def main():
    extractor = PDFDataExtractor()
    
    try:
        base_dir = Path(__file__).parent.resolve()
        input_folder = base_dir / '../2. scr/ДТ'
        output_folder = base_dir / '../2. scr/output'
        
        extractor.extract_data_from_folder(str(input_folder), str(output_folder))
        extractor.logger.info("Обработка завершена. Данные сохранены в CSV файл.")
        
    except Exception as e:
        extractor.logger.error(f"Ошибка: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()