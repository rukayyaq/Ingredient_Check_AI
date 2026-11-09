# 🥗 IngredientCheck AI

### An Intelligent Food Ingredient, Nutrition & Allergen Safety Analyzer using RAG, LLMs, Machine Learning and OCR

> Empowering consumers with transparent food data, actionable safety warnings, and AI-driven nutrition insights.

**IngredientCheck AI** is an intelligent, multimodal food analysis platform designed to help users understand packaged food ingredients, nutritional values, and potential allergens.

The application integrates **Dataset Fusion, Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Machine Learning, and Optical Character Recognition (OCR)** to deliver comprehensive food product analysis.

Built using Python and Streamlit, the application combines global food information sources with an Indian packaged food dataset to provide an interactive platform for nutritional analysis, allergen identification, product interrogation, and food label scanning.

---

## 🌐 Live Demo

🚀 **Try IngredientCheck AI:** [https://ingredientcheckai-gnemfk4ksaxrxhzqypdz3n.streamlit.app/]

> The live application may have limited functionality depending on API availability and deployment resources.

## 📸 Application Dashboard

![IngredientCheck AI Dashboard](screenshots/dashboard.png)

---

## ✨ Key Features

### 📷 1. OCR-Based Food Label Scanner

Upload an image of a packaged food label and extract its ingredient information using Tesseract OCR.

- Extracts text from uploaded food label images.
- Identifies ingredients using a custom ingredient parser.
- Detects potential allergens using rule-based matching.
- Generates food safety analysis.
- Maintains local scan history.

### 🤖 2. AI-Powered Product Analysis (LLM + RAG)

Interact with food products using an intelligent question-answering system.

- Search products from the integrated nutrition dataset.
- Retrieve relevant product-specific information.
- Generate context-aware answers using Groq's GPT-OSS-20B model.
- Supports Google Gemini as an alternative analysis model.
- Uses retrieved product information to reduce hallucinations.

### 🔗 3. Data Fusion & Quality Assessment

Combines multiple food information sources into a unified dataset.

- Integrates the existing Phase 1 dataset with Indian packaged food records.
- Standardizes product and nutritional columns.
- Handles missing values using median and categorical imputation.
- Identifies outliers using the Interquartile Range (IQR) method.
- Displays dataset quality metrics and fusion comparisons.

### 📊 4. Exploratory Data Analysis (EDA)

Provides interactive visualizations to understand food and nutrition patterns.

- Nutritional distribution analysis.
- Product price distributions.
- Serving size analysis.
- Correlation heatmaps.
- Before-and-after dataset fusion comparisons.

### 🧠 5. Machine Learning Models

Implements classification and regression models for predictive analysis.

**Classification Models:**
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Logistic Regression

**Regression Models:**
- Linear Regression
- Additional regression models for calorie prediction.

Model evaluation uses metrics such as Accuracy Score and R² Score.

### 🕘 6. Analysis History & PDF Reports

- Maintains previously scanned food label records.
- Stores analysis history locally in JSON format.
- Generates downloadable PDF food safety reports.
- Provides an organized record of previous analyses.

---

# 🧠 How IngredientCheck AI Works

IngredientCheck AI follows a hybrid architecture combining OCR, rule-based ingredient parsing, dataset retrieval, and LLM-powered analysis.

### Application Workflow

```text
                  USER INPUT
                      |
             ---------------------
             |                   |
         Food Image          Product Search
             |                   |
             ▼                   |
        Tesseract OCR             |
             |                   |
             ▼                   |
      Ingredient Extraction      |
             |                   |
             ▼                   |
       Ingredient Parser         |
             |                   |
             ▼                   |
       Allergen Detection        |
             |                   |
             └────────┬──────────┘
                      |
                      ▼
               Context Builder
                      |
                      ▼
             Integrated Food Dataset
                      |
                      ▼
              RAG Retrieval Layer
                      |
                      ▼
               Context Formation
                      |
                      ▼
               LLM Processing
                      |
              ----------------
              |              |
              ▼              ▼
           Groq API      Gemini API
              |              |
              └──────┬───────┘
                     |
                     ▼
              AI-Generated Analysis
                     |
                     ▼
            Nutrition & Safety Results
                     |
              ----------------
              |              |
              ▼              ▼
          Streamlit       PDF Report
          Dashboard
```

---

# 📚 Knowledge Base & Datasets

## 🏥 1. Fused Nutrition Dataset

The application combines food product information from multiple sources.

**Phase 1 Dataset**
- Approximately 4,959 records.
- Integrated food information from Open Food Facts, Edamam, USDA, FatSecret, and COR API sources.

**Phase 2 Dataset**
- 852 Indian packaged food records.
- Includes biscuits, snacks, chocolates, and other packaged food products.

**Final Fused Dataset**
- Approximately 5,811 records.
- 66+ columns.
- Includes nutritional values, ingredients, brands, product categories, prices, serving sizes, and data source information.

Additional fields include:

- `price_inr`
- `serving_size_g`
- `data_source`
- `energy_kcal`
- `proteins_100g`
- `fat_100g`
- `carbohydrates_100g`

## 🧬 2. Allergen Knowledge Base

The application maintains a curated allergen knowledge base containing information about common food allergens.

Supported allergen categories include:

- Peanuts
- Tree Nuts
- Milk / Dairy
- Eggs
- Soy
- Wheat / Gluten
- Fish
- Shellfish
- Sesame
- Mustard
- Celery
- Sulphites

ChromaDB stores the vector representations used for semantic retrieval of relevant allergen information.

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| UI Styling | Custom CSS |
| Primary LLM | GPT-OSS-20B via Groq |
| Alternative LLM | Google Gemini |
| RAG Framework | Retrieval-Augmented Generation |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embeddings | Sentence Transformers |
| OCR Engine | Tesseract OCR |
| OCR Integration | Pytesseract |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| PDF Generation | FPDF |
| Image Processing | Pillow (PIL) |
| Environment Management | Python-dotenv |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

---

# 🔌 External APIs & Integrations

IngredientCheck AI combines external food information sources with LLM services to provide comprehensive food analysis.

## 1. Groq API

**Purpose:** AI-powered product analysis and natural language response generation.

- Connects the application to the GPT-OSS-20B model.
- Generates contextual responses based on retrieved product information.
- Processes questions about ingredients, nutrition, and allergens.
- Uses product-specific dataset context to reduce unsupported answers.

**Model:** `openai/gpt-oss-20b`

## 2. Google Gemini API

**Purpose:** Alternative LLM-based food analysis.

- Supports AI-powered product interrogation.
- Provides an alternative model for generating responses.
- Processes food-related questions using the available application context.

**Model configured:** `gemini-2.0-flash-exp`

## 3. Open Food Facts

**Purpose:** Packaged food product information.

- Product names and brands.
- Ingredient information.
- Allergen information.
- Nutritional composition.

Official Website: https://world.openfoodfacts.org/

## 4. Edamam API

**Purpose:** Food and nutrition information.

- Nutritional values.
- Calories and macronutrients.
- Food identification and ingredient-related information.

Official Website: https://www.edamam.com/

## 5. USDA FoodData Central

**Purpose:** Nutritional composition and food reference information.

- Calories.
- Proteins.
- Fats.
- Carbohydrates.
- Other nutritional information.

Official Website: https://fdc.nal.usda.gov/

## 6. FatSecret API

**Purpose:** Food product and nutritional information.

- Food identification.
- Product descriptions.
- Nutritional information.
- Calories and macronutrients.

Official Website: https://platform.fatsecret.com/

## 7. COR API

**Purpose:** Additional food product information and product matching.

- Product identification.
- Brand information.
- Product matching scores.
- Additional food dataset enrichment.

> **Integration note:** Open Food Facts, Edamam, USDA, FatSecret, and COR are represented as food-data sources in the integrated dataset. Groq and Gemini are the LLM services used for AI response generation. API credentials should be configured only for the services actually enabled in the application.

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/rukayyaq/Ingredient_Check_AI.git

cd Ingredient_Check_AI
```

## 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv

venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv

source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Tesseract OCR

Tesseract OCR is required for the food label scanning functionality.

**Windows:**

Download and install Tesseract OCR from:

https://github.com/UB-Mannheim/tesseract/wiki

The default installation path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Linux:**

```bash
sudo apt install tesseract-ocr
```

## 5. Configure API Keys

Create a `.env` file in the root directory of the project.

Add the required credentials:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Get your API credentials:

- Groq: https://console.groq.com/
- Google AI Studio: https://aistudio.google.com/

**Important:** Never upload your `.env` file or expose API credentials on GitHub.

## 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
Ingredient_Check_AI/
│
├── .streamlit/
│   └── Streamlit configuration
│
├── api/
│   └── API integration modules
│
├── data/
│   ├── final_clean_dataset.csv
│   ├── packaged_foods_india.csv
│   └── new_fused_dataset.csv
│
├── notebooks/
│   └── Data analysis notebooks
│
├── app.py
├── config.py
├── fusion_script.py
├── clean_dataset.py
├── data_quality_check.py
├── phase2_analysis.py
├── phase2_eda_fused.py
├── ml_models_fused.py
├── llm_engine.py
├── llm_analyzer.py
├── rag_pipeline.py
├── ingredient_parser.py
├── ocr_processor.py
├── prompts.py
├── utils.py
├── check_groq.py
│
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

---

# 📊 RAG Configuration

The application uses Retrieval-Augmented Generation to provide context-aware food product analysis.

| Component | Configuration |
|---|---|
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embedding Library | Sentence Transformers |
| Vector Database | ChromaDB |
| Collection Name | allergen_knowledge |
| Retrieval Method | Semantic Similarity |
| Generation Model | GPT-OSS-20B via Groq |
| Alternative Model | Google Gemini |
| Context Strategy | Product-specific dataset context |
| Knowledge Retrieval | Allergen knowledge base |

### Why RAG?

RAG helps the application retrieve relevant information before generating an answer.

Instead of relying entirely on the language model's existing knowledge, IngredientCheck AI uses available product and allergen information to produce more relevant and context-aware responses.

---

# 📸 Screenshots

Add screenshots of your actual running application inside the `screenshots/` folder.

| Module | Screenshot |
|---|---|
| Dashboard | `screenshots/dashboard.png` |
| Data Fusion | `screenshots/fusion.png` |
| Data Quality Assessment | `screenshots/data-quality.png` |
| Exploratory Data Analysis | `screenshots/eda.png` |
| AI / RAG Integration | `screenshots/rag.png` |
| Machine Learning Models | `screenshots/ml-models.png` |
| OCR Food Label Scanner | `screenshots/ocr.png` |

Example:

```markdown
![IngredientCheck AI Dashboard](screenshots/dashboard.png)
```

---

# 🔐 Security

IngredientCheck AI follows basic security practices for API credentials and application configuration.

- API keys are stored using environment variables.
- Python-dotenv loads credentials from the local `.env` file.
- Sensitive files should be excluded through `.gitignore`.
- API credentials should never be hardcoded into source files.
- Local vector database files should not be uploaded unnecessarily.

Sensitive files include:

```text
.env
chroma_db/
venv/
__pycache__/
```

---

# ⚠️ Disclaimer

IngredientCheck AI is an academic AI project and educational prototype.

The information provided by the application is intended for informational purposes only. It is not a substitute for professional medical diagnosis, treatment, or personalized advice from a qualified healthcare professional.

Food ingredient and allergen detection may not identify every possible allergen or cross-contamination risk. Always verify product packaging and consult a qualified healthcare professional regarding serious food allergies.

---

# 🎓 Academic Information

| Field | Details |
|---|---|
| Project | IngredientCheck AI |
| Course | Large Language Models (LLM) / Data Science |
| Programme | MCA |
| Department | [SNDT Women's University P.G.Department of Computer Science] |
| University | [SNDT Women's University] |
| Academic Year | 2026 |
| Student | Rukayya Aiyoob Qureshi |

---

# 👩‍💻 Developer

**Rukayya Aiyoob Qureshi**

MCA Student | AI • Data Science • Machine Learning • Python Development

GitHub: [@rukayyaq](https://github.com/rukayyaq)

---

# 🙏 Acknowledgements

This project makes use of various open-source technologies, food information resources, and AI services.

Special thanks to:

- Open Food Facts
- Edamam
- USDA FoodData Central
- FatSecret
- Groq
- Google Gemini
- Streamlit
- ChromaDB
- Sentence Transformers
- Scikit-Learn
- Python Open-Source Community

---

# 📌 Project Status

**Status: Completed Academic Prototype**

IngredientCheck AI demonstrates the practical integration of:

- Dataset Fusion and Data Preprocessing
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- External Food Data Sources
- Semantic Search and Vector Databases
- Optical Character Recognition (OCR)
- Allergen Detection and Ingredient Parsing
- Machine Learning Classification and Regression
- Exploratory Data Analysis
- Streamlit Application Development
- Automated PDF Report Generation

---

# 📜 License

This project was developed for academic and educational purposes.

