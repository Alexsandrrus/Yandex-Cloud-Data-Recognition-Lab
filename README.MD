# DataMaster: Automated Customs Declaration Processing for EPR Compliance

## 📌 Project Overview

**DataMaster** is an intelligent document processing system that automates the extraction and analysis of goods data from Russian customs declarations (DT) to identify products subject to Extended Producer Responsibility (РОП) regulations.

## 🛠 Key Features

- **PDF Data Extraction**  
  Automated parsing of Russian-language customs declarations in PDF format
- **HS Code Recognition**  
  Identification of product categories using TN VED codes
- **EPR Compliance Filtering**  
  Matching against government regulations (Постановление №2414)
- **Smart Weight Calculations**  
  Automatic net/gross weight computations and packaging weight derivation
- **Multilingual NLP**  
  Specialized models for Russian document processing


## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Process declarations:
```bash
from scripts.pdf_processor import process_declarations

report = process_declarations(
    declarations_dir="ДТ",
    regulation_path="Постановление Правительства Российской Федерации от 29.12.2023 № 2414 (с 2024 года).pdf"
)
report.to_excel("outputs/epr_report.xlsx")
```

## 📊 Sample Output
| Product Name | HS Code Quantity |	Net Weight (kg) | Packaging Weight (kg) |
|--------------|------------------|-----------------|-----------------------|
|Детские игрушки Fisher|	9503001009 150|		450.2|	22.5|
|Пластиковые куклы|	9503002100	320|	1280.7|	64.0|

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.


