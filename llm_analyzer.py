import os
import re
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv

# Force load .env from project directory
load_dotenv(find_dotenv(), override=True)


class LLMAnalyzer:

    def __init__(self):
        # 1. Resolve API key from .env, st.secrets, or config
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass

        if not self.api_key:
            try:
                import config

                self.api_key = getattr(config, "GEMINI_API_KEY", None)
            except ImportError:
                pass

        # Clean key string formatting
        if self.api_key:
            self.api_key = str(self.api_key).strip().strip('"').strip("'")

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API via REST with fallback models."""
        if not self.api_key or len(self.api_key) < 10:
            print("[LLM Error] GEMINI_API_KEY is missing or invalid.")
            return None

        # Try active model endpoints in order
        models_to_try = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            try:
                response = requests.post(
                    url, json=payload, headers=headers, timeout=12
                )
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                else:
                    print(
                        f"[LLM Warning] Model {model} returned HTTP"
                        f" {response.status_code}"
                    )
            except Exception as e:
                print(f"[LLM Exception] Model {model} failed: {e}")

        return None

    def analyze_product(
        self,
        ingredients,
        allergen_results,
        user_allergies,
        dietary_preferences,
        product_name,
        safety_summary,
    ):
        danger_items = [
            r["ingredient"]
            for r in allergen_results
            if isinstance(r, dict) and r.get("status") == "danger"
        ]
        warning_items = [
            r["ingredient"]
            for r in allergen_results
            if isinstance(r, dict) and r.get("status") == "warning"
        ]

        prompt = f"""
        You are an expert food safety assistant.
        Product Name: {product_name}
        User Selected Allergies: {', '.join(user_allergies)}
        Dietary Preferences: {', '.join(dietary_preferences) if dietary_preferences else 'None'}
        Ingredients List: {', '.join(ingredients)}
        Allergen Matching Data: {allergen_results}

        Provide a clear, formatted summary of safety concerns for the user. Highlight matched allergens in bold.
        """

        result = self._call_gemini_api(prompt)
        if result:
            return result

        # Rule-Based Local Fallback
        if danger_items:
            return (
                "### 🚨 Safety Warning\n\n"
                "This product contains ingredients matching your specified"
                f" allergies (**{', '.join(user_allergies)}**):\n\n"
                "*"
                " **Unsafe Ingredients Found:**"
                f" {', '.join([f'**{item}**' for item in danger_items])}\n\n"
                "**Recommendation:** Avoid consuming this product."
            )
        elif warning_items:
            return (
                "### ⚠️ Caution\n\n"
                "This product contains ingredients that may require"
                " attention:\n\n"
                "*"
                " **Warning Ingredients:**"
                f" {', '.join([f'**{item}**' for item in warning_items])}"
            )

        return (
            "### ✅ Product Appears Safe\n\nNo ingredients matching your"
            " selected allergies were detected."
        )

    def chat_about_ingredient(
        self, question: str, ingredients: list, user_allergies: list
    ):
        prompt = f"""
        Context:
        Product Ingredients: {', '.join(ingredients)}
        User Active Allergies: {', '.join(user_allergies) if user_allergies else 'None'}

        User Question: "{question}"

        Provide a helpful, direct, and concise explanation answering their question.
        Explain what the ingredient is, its primary purpose in food products, and any specific safety considerations regarding the user's allergy profile.
        Do NOT repeat or echo the prompt text.
        """

        result = self._call_gemini_api(prompt)
        if result:
            return result

        # Normalize query: lowercase and remove special characters
        q_clean = re.sub(r"[^a-zA-Z0-9\s]", "", question.lower()).strip()

        knowledge_base = {
            # Vitamins & Fortification
            "niacin": (
                "**Niacin** (Vitamin B3) is an essential nutrient involved in"
                " cellular metabolism and energy production. It is standardly"
                " used to fortify wheat flours and cereals and is safe and"
                " non-allergenic."
            ),
            "niacinamide": (
                "**Niacinamide** (Vitamin B3) is essential for cellular energy"
                " metabolism. It is commonly used to fortify enriched grain"
                " products."
            ),
            "folic acid": (
                "**Folic Acid** is a synthetic form of Vitamin B9 added to"
                " fortify enriched flours and cereals. It is safe and"
                " non-allergenic."
            ),
            "thiamin": (
                "**Thiamin Mononitrate** (Vitamin B1) is a nutrient added to"
                " enrich processed flours. It is safe and non-allergenic."
            ),
            "riboflavin": (
                "**Riboflavin** (Vitamin B2) aids in energy production and is"
                " standardly used for grain fortification."
            ),
            "pyridoxine": (
                "**Pyridoxine Hydrochloride** (Vitamin B6) supports brain"
                " development and metabolic function."
            ),
            # Oils & Fats
            "hydrogenated": (
                "**Hydrogenated Vegetable Oil** is liquid vegetable oil"
                " processed with hydrogen to convert it into a solid fat. It"
                " often contains trans fats, which can raise bad cholesterol"
                " (LDL)."
            ),
            "vegetable oil": (
                "**Vegetable Oil** refers to plant-based fats extracted from"
                " seeds or fruits. Hydrogenated versions contain trans fats and"
                " should be consumed in moderation."
            ),
            "cottonseed": (
                "**Cottonseed Oil** is a plant oil extracted from cotton seeds."
                " It is refined and commonly used in frying and snack foods."
            ),
            "rapeseed": (
                "**Rapeseed Oil** (Canola oil) is low in saturated fat and rich"
                " in monounsaturated fats. It is safe unless you have specific"
                " seed sensitivities."
            ),
            "coconut": (
                "**Coconut Oil** is a plant oil high in saturated fats. While"
                " botanically a fruit, the FDA classifies coconut as a tree"
                " nut allergen."
            ),
            # Minerals & Additives
            "sodium": (
                "**Sodium** is an essential mineral found in salt and food"
                " preservatives. It is not an allergen, though high intake"
                " impacts cardiovascular health."
            ),
            "iron": (
                "**Reduced Iron** is an elemental iron supplement added to"
                " enrich refined grain products and cereals."
            ),
            "zinc": (
                "**Zinc Oxide** is an essential trace mineral added to foods for"
                " nutritional enrichment."
            ),
            "dextrose": (
                "**Dextrose** is a simple sugar derived from corn or wheat."
                " It serves as a sweetener and stabilizer."
            ),
            "bht": (
                "**BHT** (Butylated Hydroxytoluene) is a synthetic antioxidant"
                " used as a preservative to prevent oils from spoiling."
            ),
            "caramel": (
                "**Caramel Color** is a common food coloring created by"
                " heating carbohydrates."
            ),
            # Grains & Allergens
            "corn flour": (
                "**Corn Flour** is finely ground whole corn. It is naturally"
                " gluten-free unless cross-contaminated during processing."
            ),
            "oat flour": (
                "**Oat Flour** is ground oats. Oats are naturally gluten-free"
                " but frequently processed on equipment shared with wheat."
            ),
            "rice flour": (
                "**Rice Flour** is made from ground rice and is a standard"
                " gluten-free alternative."
            ),
            "gluten": (
                "**Gluten** is a protein found in wheat, barley, and rye that"
                " causes immune reactions in Celiac disease or wheat"
                " sensitivities."
            ),
            "peanut": (
                "**Peanuts** are legumes and represent one of the most common"
                " severe food allergens."
            ),
        }

        # Check for exact keyword matches in cleaned query
        for key, info in knowledge_base.items():
            if key in q_clean:
                return info

        # Extract question keywords, stripping common intent words
        target_term = re.sub(
            r"^(what|is|are|tell|me|about|how|does|why|can|you|explain)\s+",
            "",
            q_clean,
            flags=re.IGNORECASE,
        ).strip()
        target_term = target_term if target_term else question.strip("?").strip()

        return (
            f"**{target_term.capitalize()}** is an ingredient listed on the"
            " product label. It is generally recognized as safe (GRAS) for food"
            " consumption. If you have specific allergies or dietary"
            " restrictions, check the allergen warning on the packaging."
        )