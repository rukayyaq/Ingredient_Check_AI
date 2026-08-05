import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (SECURE: Read from environment variables, do not hardcode)
# Set GEMINI_API_KEY="your_key_here" inside a .env file in the same folder
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("ERROR: GEMINI_API_KEY is missing. Please set it in your .env file.")

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Vector Database
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "allergen_knowledge"

# LLM Settings
LLM_MODEL = "gemini-2.0-flash-exp"

# App Settings
APP_TITLE = "🍽️ IngredientCheck AI"
APP_SUBTITLE = "RAG-Powered Food Label Allergen Analyzer"
MAX_FILE_SIZE_MB = 10

# Allergen Categories
ALLERGEN_CATEGORIES = [
    "Peanuts", "Tree Nuts", "Milk/Dairy", "Eggs",
    "Soy", "Wheat/Gluten", "Fish", "Shellfish",
    "Sesame", "Mustard", "Celery", "Sulphites"
]

# Dietary Preferences
DIETARY_PREFERENCES = [
    "Vegan", "Vegetarian", "Halal", "Kosher",
    "Jain", "Gluten-Free", "Lactose-Free"
]