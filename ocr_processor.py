# ocr_processor.py
import io
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import streamlit as st
import pytesseract  # Edited: Switched from EasyOCR to Tesseract
import config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Edited: Removed the massive EasyOCR download. 
# Streamlit Cloud uses Linux. We just tell it where Tesseract is installed.
@st.cache_resource(show_spinner="Loading OCR Engine...")
def get_ocr_engine():
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return pytesseract

class OCRProcessor:
    def __init__(self):
        try:
            self.engine = get_ocr_engine()
        except Exception:
            self.engine = None

    def extract_text_from_image(self, image_input) -> Tuple[Optional[str], str]:
        try:
            if isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, str):
                img = Image.open(image_input)
            else:
                img = image_input

            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Edited: Switched to Tesseract's image_to_string function
            if self.engine:
                extracted_text = self.engine.image_to_string(img)

                if extracted_text and len(extracted_text.strip()) > 3:
                    return extracted_text.strip(), "Analyzed Product"

            return "NOT_FOUND: No readable text detected.", "Error"

        except Exception as e:
            return f"CRITICAL ERROR: {str(e)}", "Error"