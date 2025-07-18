# Intelligent-Descriptive-Answer-Grading-System
# 🧠 Smart Answer Evaluator

This Streamlit app evaluates a student's answer against a model answer by extracting text from PDFs/images and computing similarity metrics like Cosine, Jaccard, Bigram, Synonym similarity, and Grammar error count.

## 🚀 Features
- Upload student and model answers (PDF or Image).
- Extracts text using OCR (Tesseract).
- Computes multiple text similarity metrics.
- Detects grammar errors.
- Gives a final score with performance level.

## 📦 Requirements
- Python 3.7+
- Tesseract OCR installed on your system
  - Windows: [Download Tesseract](https://github.com/tesseract-ocr/tesseract)
  - Linux/Mac: Install via package manager (`sudo apt install tesseract-ocr` or `brew install tesseract`)

## 🔧 Installation
```bash
git clone https://github.com/yourusername/smart-answer-evaluator.git
cd smart-answer-evaluator
pip install -r requirements.txt
