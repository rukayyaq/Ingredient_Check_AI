import io
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import streamlit as st

# Suppress PyTorch and EasyOCR deprecation/execution warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import easyocr

# Cache the EasyOCR model so it loads into memory only once
@st.cache_resource(show_spinner="Loading OCR Model...")
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

class OCRProcessor:
    def __init__(self):
        try:
            self.reader = get_ocr_reader()
        except Exception:
            self.reader = None

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

            img_np = np.array(img)

            if self.reader:
                results = self.reader.readtext(img_np, detail=0)
                extracted_text = "\n".join(results)

                if extracted_text and len(extracted_text.strip()) > 3:
                    return extracted_text.strip(), "Analyzed Product"

            return "NOT_FOUND: No readable text detected.", "Error"

        except Exception as e:
            return f"CRITICAL ERROR: {str(e)}", "Error"