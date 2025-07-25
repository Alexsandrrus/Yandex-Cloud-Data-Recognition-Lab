# Проект: Извлечение данных из PDF-документов
**Этот проект предоставляет инструменты для извлечения текста и структурированных данных из PDF-документов с использованием различных методов, включая OCR и парсинг текста.**


## Особенности проекта
**Извлечение текста из PDF с использованием нескольких методов:**

- PDFPlumber

- PyPDF2

- PDFMiner

- Tesseract OCR

- Автоматический выбор лучшего метода для каждого документа

**Извлечение структурированных данных из деклараций на товары:**

- Маркировка и количество

- Код товара (ТН ВЭД)

- Вес брутто и нетто

**Генерация отчетов в форматах CSV и PDF**

**Улучшение качества изображений для OCR**

**Логирование всех операций**

## Структура проекта

```bash
├── 1. docs/              # Папка с исходными документами
├── 2. scr/               # Папка со скриптами
│   ├── debug/            # Папка для отладочной информации
│   ├── djsans/           # Папка со шрифтами для генерации PDF
│   ├── ДТ/               # Папка с PDF-файлами деклараций на товары
│   ├── output/           # Папка для результатов
│   ├── get_dt.py         # Скрипт для извлечения данных из ДТ
│   └── get_TNVED.py      # Скрипт для извлечения текста из PDF
├── .venv/                # Виртуальное окружение
├── README.MD             # Этот файл
├── requirements.txt      # Зависимости проекта
└── pdf_data_extractor.log # Лог-файл
```

## Установка и запуск

**Предварительные требования**:

1. Установите Tesseract OCR

2. Установите Poppler (для работы с PDF)

3. Убедитесь, что русский языковой пакет для Tesseract (rus.traineddata) установлен в папку Tesseract tessdata

**Установка зависимостей**
```bash
pip install -r requirements.txt
```

## Запуск скриптов

**Извлечение данных из деклараций на товары (ДТ):**

```bash
python get_dt.py
```
1. Поместите PDF-файлы ДТ в папку 2. scr/ДТ

2. Результаты будут сохранены в 2. scr/output/extracted_data.csv и extracted_data.pdf

**Извлечение Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года) из PDF:**

```bash
python get_TNVED.py
```

1. Укажите путь к PDF-файлу в скрипте

2. Текст будет сохранен в 2. scr/output/extracted_text.txt

## Настройка

**При необходимости измените пути в коде:**

1. tesseract_cmd - путь к Tesseract OCR

2. poppler_path - путь к Poppler

3. font_dir - путь к папке со шрифтами

## Зависимости

**Основные зависимости (см. requirements.txt):**

- text
- PyPDF2
- pdfminer.six
- pdfplumber
- pdf2image
- pytesseract
- Pillow
- fpdf

## Логирование
Все операции логируются в файл pdf_data_extractor.log с указанием времени, уровня сообщения и деталей операции.

## Особенности реализации

1. Автоматический выбор лучшего метода извлечения текста

2. Улучшение изображений для OCR (контраст, резкость, бинаризация)

3. Гибкие регулярные выражения для извлечения структурированных данных

4. Поддержка кириллицы в отчетах PDF

5. Обработка ошибок на всех этапах работы

## Примеры использования

1. Обработка партии деклараций на товары

2. Извлечение текста из нормативных документов

3. Анализ структурированных данных из PDF-форм