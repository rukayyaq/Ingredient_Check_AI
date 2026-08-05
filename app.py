import os
import tempfile
from datetime import datetime
import pandas as pd
from PIL import Image
import streamlit as st

import config
from ingredient_parser import IngredientParser
from llm_analyzer import LLMAnalyzer
from ocr_processor import OCRProcessor
from rag_pipeline import RAGPipeline
from utils import create_download_report, init_session_state

# Page config
st.set_page_config(
    page_title=getattr(config, "APP_TITLE", "IngredientCheck AI"),
    page_icon="🥫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

# Title
st.title(f"🥫 {getattr(config, 'APP_TITLE', 'IngredientCheck AI')}")
st.markdown(
    f"*{getattr(config, 'APP_SUBTITLE', 'RAG-Powered Allergen Detection')}*"
)
st.markdown("---")

# Sidebar - User Profile
with st.sidebar:
    st.header("👤 Your Allergy Profile")

    user_allergies = st.multiselect(
        "Select your allergies:",
        getattr(
            config,
            "ALLERGEN_CATEGORIES",
            [
                "Peanuts",
                "Tree Nuts",
                "Milk",
                "Eggs",
                "Soy",
                "Wheat",
                "Gluten",
                "Fish",
                "Shellfish",
                "Sesame",
            ],
        ),
        help="Select all allergens that apply to you",
    )

    dietary_preferences = st.multiselect(
        "Dietary Preferences (Optional):",
        getattr(
            config,
            "DIETARY_PREFERENCES",
            ["Vegan", "Vegetarian", "Halal", "Kosher"],
        ),
        help="Select any dietary preferences",
    )

    st.markdown("---")

    st.header("ℹ️ About")
    st.markdown("""
    **IngredientCheck AI** uses:
    - 🔍 OCR to read food labels
    - 🧠 RAG to retrieve allergen info
    - 🤖 LLM to analyze and explain
    
    **Always consult a doctor** for serious allergy concerns.
    """)

    st.markdown("---")

    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📸 Scan Label", "💬 Ask Questions", "📜 History"])

# Tab 1: Scan Label
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Food Label")

        uploaded_file = st.file_uploader(
            "Upload a food label image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Take a clear photo of the ingredients list",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image, caption="Uploaded Label", use_container_width=True
            )

            if st.button(
                "🔍 Analyze Ingredients",
                type="primary",
                use_container_width=True,
            ):
                if not user_allergies:
                    st.warning(
                        "⚠️ Please select your allergies in the sidebar first!"
                    )
                else:
                    tmp_path = None
                    try:
                        with st.spinner("🔄 Processing..."):
                            # Step 1: Write file and explicitly close handle before processing
                            file_ext = (
                                os.path.splitext(uploaded_file.name)[1]
                                or ".jpg"
                            )
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=file_ext
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            st.info("📸 Extracting text from image...")
                            ocr = OCRProcessor()

                            # Support both path-based and image-based OCR
                            try:
                                (
                                    ingredient_text,
                                    product_name,
                                ) = ocr.extract_text_from_image(tmp_path)
                            except Exception:
                                (
                                    ingredient_text,
                                    product_name,
                                ) = ocr.extract_text_from_image(image)

                            if (
                                not ingredient_text
                                or ingredient_text == "NOT_FOUND"
                            ):
                                st.error(
                                    "❌ No ingredients list detected. Try a"
                                    " clearer photo."
                                )
                                st.stop()
                            else:
                                st.success(f"✅ Product: {product_name}")

                                # Step 2: Parse Ingredients
                                st.info("✂️ Parsing ingredients...")
                                parser = IngredientParser()
                                ingredients = parser.parse_ingredients(
                                    ingredient_text
                                )

                                if not ingredients:
                                    st.error(
                                        "❌ Parser couldn't split ingredients"
                                        " from text."
                                    )
                                    st.stop()

                                st.success(
                                    f"✅ Found {len(ingredients)} ingredients"
                                )

                                # Step 3: Check Allergens
                                st.info(
                                    "🔎 Checking against your allergy"
                                    " profile..."
                                )
                                allergen_results = parser.check_allergens(
                                    ingredients, user_allergies
                                )
                                safety_summary = parser.get_safety_summary(
                                    allergen_results
                                )

                                # Step 4: LLM Analysis
                                st.info("🤖 Generating AI analysis...")
                                analyzer = LLMAnalyzer()
                                analysis = analyzer.analyze_product(
                                    ingredients,
                                    allergen_results,
                                    user_allergies,
                                    dietary_preferences,
                                    product_name,
                                    safety_summary,
                                )

                                # Store in session state
                                st.session_state.current_analysis = analysis
                                st.session_state.current_product = product_name
                                st.session_state.current_ingredients = (
                                    ingredients
                                )
                                st.session_state.current_allergen_results = (
                                    allergen_results
                                )
                                st.session_state.current_safety_summary = (
                                    safety_summary
                                )

                                st.success("✅ Analysis complete!")
                                st.rerun()

                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

    with col2:
        st.subheader("📊 Analysis Results")

        if st.session_state.get("current_analysis"):
            # Safety verdict
            if st.session_state.get("current_safety_summary"):
                summary = st.session_state.current_safety_summary

                if summary["is_safe"]:
                    st.success("### ✅ SAFE - No allergens detected!")
                else:
                    st.error(
                        "### 🚨 DANGER - Allergen Detected!"
                        f" ({summary['danger']} found)"
                    )

                # Metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("🔴 Dangerous", summary["danger"])
                col_b.metric("🟡 Warnings", summary["warning"])
                col_c.metric("🟢 Safe", summary["safe"])

            st.markdown("---")
            # Analysis text
            st.markdown(st.session_state.current_analysis)

            st.write("")
            # Download button with explicitly mapped parameters
            if st.button("📥 Download Report (PDF)", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_path = create_download_report(
                        product_name=st.session_state.get(
                            "current_product", "Food Product"
                        ),
                        user_allergies=user_allergies,
                        dietary_prefs=dietary_preferences,
                        analysis_summary=st.session_state.get(
                            "current_analysis", ""
                        ),
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "💾 Click to Download",
                            f,
                            file_name=(
                                "ingredientcheck_report_"
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            ),
                            mime="application/pdf",
                            use_container_width=True,
                        )
        else:
            st.info(
                "👈 Upload a food label image and click 'Analyze Ingredients'"
                " to see results here."
            )

            with st.expander("📖 How it works", expanded=True):
                st.markdown("""
                ### 🔬 Process Flow:
                1. **Upload** a photo of any food product's ingredient list
                2. **OCR** extracts the text from the image
                3. **Parser** splits ingredients into individual items
                4. **RAG Pipeline** retrieves allergen knowledge from database
                5. **LLM** analyzes against your allergy profile
                6. **Results** show safe/dangerous/warning ingredients
                """)

# Tab 2: Chat
with tab2:
    st.subheader("💬 Ask About Ingredients")

    if st.session_state.get("current_ingredients"):
        st.info(
            "📝 Current product ingredients:"
            f" {len(st.session_state.current_ingredients)} items"
        )

        user_question = st.text_input(
            "Ask about any ingredient:",
            placeholder=(
                "E.g., Is maltodextrin safe for gluten allergy? What is sodium"
                " caseinate?"
            ),
            key="chat_input",
        )

        if st.button("🔍 Ask", type="primary") and user_question:
            with st.spinner("🤖 Thinking..."):
                analyzer = LLMAnalyzer()
                response = analyzer.chat_about_ingredient(
                    user_question,
                    st.session_state.current_ingredients,
                    user_allergies,
                )

                # Add to chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })

                st.rerun()

        # Display chat history
        if st.session_state.get("chat_history"):
            st.markdown("---")
            st.subheader("📜 Chat History")

            for chat in reversed(st.session_state.chat_history[-10:]):
                with st.chat_message("user"):
                    st.markdown(
                        f"**{chat['timestamp']}** - {chat['question']}"
                    )
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
    else:
        st.warning(
            "⚠️ Please scan a product label first in the 'Scan Label' tab."
        )

# Tab 3: History
with tab3:
    st.subheader("📜 Scan History")

    if st.session_state.get("current_product"):
        st.markdown(
            f"**Last Scanned Product:** {st.session_state.current_product}"
        )
        st.markdown(
            "**Ingredients Found:**"
            f" {len(st.session_state.current_ingredients)}"
        )

        if st.session_state.get("current_allergen_results"):
            data = []
            for r in st.session_state.current_allergen_results:
                data.append({
                    "Ingredient": r["ingredient"],
                    "Status": r["status"].upper(),
                    "Allergen Group": r["allergen_group"] or "-",
                    "Risk Level": r["risk_level"] or "-",
                })

            df = pd.DataFrame(data)

            def color_status(val):
                if val == "DANGER":
                    return (
                        "background-color: #ffcccc; color: #900000; font-weight:"
                        " bold;"
                    )
                elif val == "WARNING":
                    return "background-color: #fff3cd; color: #856404;"
                else:
                    return "background-color: #d4edda; color: #155724;"

            st.dataframe(
                df.style.map(color_status, subset=["Status"]),
                use_container_width=True,
            )
    else:
        st.info("No scans yet. Upload a label to get started!")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>🥫 IngredientCheck AI | RAG-Powered Food Safety | For"
    " informational purposes only</small></center>",
    unsafe_allow_html=True,
)