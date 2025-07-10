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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self):
        self.cache = {}
        
    def extract_text(self, pdf_path):
        """Extract text from PDF using combined methods"""
        if pdf_path in self.cache:
            return self.cache[pdf_path]
            
        try:
            text = self._extract_with_pdfplumber(pdf_path)
            
            # Fallback to OCR if text extraction is poor
            if len(' '.join(text)) < 100:
                ocr_text = self._extract_with_ocr(pdf_path)
                if len(ocr_text) > len(text):
                    text = ocr_text
                    
            self.cache[pdf_path] = text
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []

    def _extract_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber"""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if not page_text:
                        page_text = " ".join([word['text'] for word in page.extract_words()])
                    text.append(page_text)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
        return text

    def _extract_with_ocr(self, pdf_path):
        """Extract text using OCR"""
        text = []
        try:
            images = self._convert_pdf_to_images(pdf_path)
            for img in images:
                processed = self._preprocess_image(img)
                page_text = pytesseract.image_to_string(processed, lang='rus+eng')
                text.append(page_text)
        except Exception as e:
            logger.error(f"OCR error: {e}")
        return text

    def _convert_pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF to images"""
        try:
            from pdf2image import convert_from_path
            return convert_from_path(pdf_path, dpi=dpi)
        except ImportError:
            logger.error("Please install pdf2image: pip install pdf2image")
            return []
        except Exception as e:
            logger.error(f"PDF to image conversion error: {e}")
            return []

    def _preprocess_image(self, image):
        """Enhance image for better OCR results"""
        try:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return Image.fromarray(cleaned)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return image

class ROPProcessor:
    def __init__(self):
        logger.info("Initializing ROP processor")
        self.extractor = PDFTextExtractor()

    def load_rop_reference(self, pdf_path):
        """Load HS codes from reference PDF"""
        if not os.path.exists(pdf_path):
            logger.error(f"Reference file not found: {pdf_path}")
            return []
        
        try:
            text_pages = self.extractor.extract_text(pdf_path)
            full_text = " ".join(text_pages)
            
            # Save extracted text for debugging
            debug_dir = os.path.join("debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, "extracted_text.txt")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # Improved patterns for HS code detection
            patterns = [
                r'\b\d{4}[\.\s]?\d{2}[\.\s]?\d{2,4}\b',  # 10-digit codes with separators
                r'\b\d{6,10}\b',  # 6-10 digit codes
                r'ТН\s*ВЭД\s*[№]?\s*(\d{6,10})',  # Russian HS code notation
                r'код\s*ТН\s*ВЭД\s*[№]?\s*(\d{6,10})'
            ]
            
            found_codes = set()
            for pattern in patterns:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    code = match.group(1) if match.groups() else match.group()
                    code = re.sub(r'[^\d]', '', code)
                    if 6 <= len(code) <= 10:
                        found_codes.add(code)
            
            if not found_codes:
                found_codes.update(self._find_codes_in_tables(pdf_path))
            
            logger.info(f"Found {len(found_codes)} HS codes in reference")
            return sorted(found_codes)
            
        except Exception as e:
            logger.error(f"Error loading reference codes: {e}")
            return []

    def _find_codes_in_tables(self, pdf_path):
        """Extract HS codes from PDF tables"""
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
            logger.error(f"Table extraction error: {e}")
        return codes

    def extract_goods_info(self, pdf_path):
        """Extract goods information from declaration"""
        try:
            text_pages = self.extractor.extract_text(pdf_path)
            full_text = " ".join(text_pages)
            
            # Improved pattern for goods extraction
            goods_pattern = (
                r'(?P<hs_code>\d{6,10})'  # HS code
                r'(?P<description>.+?)'  # Description
                r'(?P<quantity>\d+[\.,]?\d*)'  # Quantity
                r'\s*(?P<unit>кг|шт|тн|л)'  # Unit
            )
            
            goods_data = []
            for match in re.finditer(goods_pattern, full_text, re.DOTALL):
                goods_data.append({
                    'hs_code': match.group('hs_code'),
                    'description': re.sub(r'\s+', ' ', match.group('description').strip()),
                    'quantity': float(match.group('quantity').replace(',', '.')),
                    'unit': match.group('unit')
                })
                
            return goods_data
            
        except Exception as e:
            logger.error(f"Error extracting goods from {pdf_path}: {e}")
            return []

    def process_folder(self, input_dir, reference_path, output_path):
        """Process all PDFs in input directory"""
        try:
            # Verify paths
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory not found: {input_dir}")
            if not os.path.exists(reference_path):
                raise FileNotFoundError(f"Reference file not found: {reference_path}")
            
            # Load reference codes
            rop_codes = self.load_rop_reference(reference_path)
            if not rop_codes:
                logger.error("No HS codes found in reference document")
                return None
                
            # Get PDF files
            pdf_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            if not pdf_files:
                logger.error("No PDF files found in input directory")
                return None
                
            # Process files
            results = []
            for pdf_file in tqdm(pdf_files, desc="Processing declarations"):
                goods = self.extract_goods_info(pdf_file)
                for good in goods:
                    if any(good['hs_code'].startswith(code) for code in rop_codes):
                        results.append({
                            'File': os.path.basename(pdf_file),
                            'HS Code': good['hs_code'],
                            'Description': good['description'],
                            'Quantity': good['quantity'],
                            'Unit': good['unit']
                        })
            
            # Save results
            if results:
                df = pd.DataFrame(results)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_excel(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
                return df
            else:
                logger.warning("No matching goods found in any declarations")
                return None
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None

def main():
    try:
        # Initialize processor
        processor = ROPProcessor()
        
        # Configure paths - adjust these to your actual paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        paths = {
            'input': os.path.normpath(os.path.join(base_dir, "2. scr", "ДТ")),
            'reference': os.path.normpath(os.path.join(base_dir, "1. docs", "Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf")),
            'output': os.path.normpath(os.path.join(base_dir, "2. scr", "output", "results.xlsx"))
        }
        
        # Process files
        result = processor.process_folder(
            paths['input'],
            paths['reference'],
            paths['output']
        )
        
        if result is not None:
            print("\nProcessing completed successfully!")
            print(f"Total records processed: {len(result)}")
            print(f"Results saved to: {paths['output']}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()