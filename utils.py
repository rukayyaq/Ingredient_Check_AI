import re
import streamlit as st
from fpdf import FPDF


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "parsed_ingredients" not in st.session_state:
        st.session_state.parsed_ingredients = []
    if "allergen_results" not in st.session_state:
        st.session_state.allergen_results = []


def sanitize_ascii(text) -> str:
    """Converts any input safely to pure ASCII string to prevent FPDF unicode errors."""
    if not text:
        return ""
    return str(text).encode('ascii', 'ignore').decode('ascii').strip()


def create_download_report(product_name, user_allergies, dietary_prefs, analysis_summary, filepath="allergen_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "IngredientCheck AI - Safety Analysis Report", ln=True, align='C')
    pdf.ln(5)
    
    # Product Details
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 7, f"Product Name: {sanitize_ascii(product_name)}", ln=True)
    
    pdf.set_font("Helvetica", '', 10)
    allergies_str = ", ".join(user_allergies) if user_allergies else "None"
    diet_str = ", ".join(dietary_prefs) if dietary_prefs else "None"
    
    pdf.cell(0, 6, f"User Active Allergies: {sanitize_ascii(allergies_str)}", ln=True)
    pdf.cell(0, 6, f"Dietary Preferences: {sanitize_ascii(diet_str)}", ln=True)
    pdf.ln(5)
    
    # Summary Header
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "Safety Summary", ln=True)
    pdf.set_font("Helvetica", '', 10)
    
    # Force convert analysis_summary to string safely
    if isinstance(analysis_summary, dict):
        summary_text = "\n".join([f"{k}: {v}" for k, v in analysis_summary.items()])
    else:
        summary_text = str(analysis_summary)
    
    # Perform string operations on clean string
    cleaned_summary = summary_text.replace("**", "").replace("###", "").strip()
    
    pdf.multi_cell(0, 6, sanitize_ascii(cleaned_summary))
    
    pdf.output(filepath)
    return filepath