# 🥫 IngredientCheck AI

RAG-Powered Food Label Allergen Analyzer using LLMs

## 🎯 Problem
Food labels use scientific names that hide common allergens.
250M+ people worldwide have food allergies.

## 🔬 How It Works
1. 📸 **OCR** - Extract text from food label images
2. ✂️ **Parse** - Split into individual ingredients  
3. 🧮 **Embed** - Convert to vectors using BAAI/bge-small-en
4. 🔍 **Retrieve** - Search ChromaDB for allergen matches
5. 🤖 **Analyze** - LLM generates personalized safety report

## 🛠️ Tech Stack
- **OCR:** Google Gemini Vision
- **Embeddings:** BAAI/bge-small-en-v1.5
- **Vector DB:** ChromaDB
- **LLM:** Gemini 2.0 Flash
- **Frontend:** Streamlit

## 🚀 Quick Start

### 1. Clone & Install
```bash
cd ingredientcheck-ai
pip install -r requirements.txt