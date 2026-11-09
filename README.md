# 🥗 IngredientCheck AI

### An Intelligent Food Ingredient, Nutrition & Allergen Safety Analyzer using RAG, LLMs, and OCR

> **Empowering consumers with transparent food data, actionable safety warnings, and AI-driven nutrition insights.**

**IngredientCheck AI** is an advanced, multimodal data platform designed to analyze packaged food products. It combines **Dataset Fusion** (merging global APIs with an Indian packaged foods dataset), **Retrieval-Augmented Generation (RAG)**, **Machine Learning**, and **Tesseract OCR** to provide deep nutritional analysis, allergen detection, and predictive modeling.

The application features a custom-themed **Streamlit** interface that handles everything from data quality assessment and exploratory data analysis to live LLM product interrogation and label scanning.

* * *

## ✨ Key Features

### 📷 OCR Label Scanner
Upload an image of a food label. Tesseract OCR extracts the text, parses the ingredients, and instantly flags potential allergens based on your personalized profile.

### 🤖 AI Product Analysis (LLM + RAG)
Select any product from the fused dataset and ask questions. The system retrieves the exact context (nutrition, ingredients, price) and uses Groq's `gpt-oss-20b` (with Gemini fallback) to generate accurate, hallucination-free responses.

### 🔗 Data Fusion & Quality Assessment
The backend merges a 4,959-row Phase 1 dataset with a new 852-row Indian Packaged Foods dataset. The UI includes a dedicated data quality dashboard (Part D) highlighting missing values, outliers (via IQR), and API-added columns.

### 📈 Comprehensive EDA
Interactive visualizations including price distributions, serving size boxplots, correlation heatmaps, and before/after fusion comparisons.

### 🧠 Machine Learning Models (Task 2)
Includes 5 Classification models (predicting high fat) and 3 Regression models (predicting calories), evaluating performance using `accuracy_score` and `r2_score`.

### 🕘 Analysis History & PDF Reports
Every scanned label is saved to a local JSON history. Users can download professional PDF safety reports generated on the fly.

* * *

# 🧠 How IngredientCheck AI Works

IngredientCheck AI follows a hybrid approach, utilizing rule-based parsing for quick allergen checks and LLM/RAG for deep, context-aware product analysis.

                   User Action (Scan Image or Search Product)
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Tesseract OCR Engine  │ (If scanning an image)
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Ingredient Parser &    │ (Regex-based extraction)
                         │  Allergen Rule Engine   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Context Builder        │ (Extracts row from Fused CSV)
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Groq API (Llama/GPT)   │ (System prompt prevents 
                         └────────────┬────────────┘  hallucination based on CSV)
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  ChromaDB (Vector DB)   │ (Retrieves allergen knowledge)
                         └─────────────────────────┘

* * *

# 📚 Knowledge Base & Datasets

## 🏥 Fused Nutrition Dataset
The core of the application is a highly integrated dataset:
*   **Phase 1 Data:** ~4,959 records (OpenFoodFacts, Edamam, USDA, FatSecret, COR API).
*   **API Data:** 852 records (Indian Packaged Foods - biscuits, snacks, chocolates).
*   **Fused Dataset:** ~5,811 records, 66+ columns, including API-specific columns like `price_inr`, `serving_size_g`, and `data_source`.

## 🧬 Allergen Knowledge Base
A curated database containing allergen rules (Peanuts, Tree Nuts, Milk, Egg, Gluten, Soy, Fish, Shellfish, Sesame) and their scientific names, stored in ChromaDB for semantic retrieval.

* * *

# 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Programming Language** | Python |
| **Frontend** | Streamlit (Custom CSS dark theme) |
| **Primary LLM** | Groq (`openai/gpt-oss-20b`) |
| **Fallback LLM** | Google Gemini (`gemini-2.0-flash`) |
| **Vector Database** | ChromaDB |
| **Embeddings** | BAAI/bge-small-en-v1.5 via Sentence Transformers |
| **OCR Engine** | Tesseract OCR (`pytesseract`) |
| **Machine Learning** | Scikit-Learn (Naive Bayes, SVM, Decision Tree, Random Forest, Logistic/Linear Regression) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **PDF Generation** | FPDF |
| **Image Processing** | Pillow (PIL) |

* * *

# 🚀 Quick Start

## 1. Clone the Repository
```bash
git clone https://github.com/rukayyaq/Ingredient_Check_AI.git
cd Ingredient_Check_AI