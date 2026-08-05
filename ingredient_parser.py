# ingredient_parser.py
import re

class IngredientParser:
    def __init__(self):
        # Explicit non-allergen terms to prevent false positives
        self.harmless_terms = {
            "iron", "reduced iron", "barley", "malted barley", "folic acid",
            "thiamin", "riboflavin", "niacin", "ascorbic acid", "salt", "sugar",
            "corn", "cornmeal", "flour", "wheat", "wheat flour", "rice", "oil",
            "soybean oil", "vegetable oil", "honey", "natural flavor"
        }

    def clean_text(self, text: str) -> str:
        """Clean raw OCR text before parsing."""
        if not text:
            return ""

        # Remove 'Ingredients:' prefix if present
        text = re.sub(r'^(ingredients|contains)[\s:]*', '', text, flags=re.IGNORECASE)
        
        # Remove footnotes like "*One of the B vitamins"
        text = re.sub(r'\*+.*?(?=(\n|$))', '', text)
        
        # Normalize whitespace and line breaks
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def parse_ingredients(self, raw_text: str) -> list:
        """Parses complex ingredient lists and splits advisory statements properly."""
        
        # ==========================================
        # 🚨 DEBUGGING BLOCK: SHOW THE RAW TEXT 🚨
        # ==========================================
        # This prints the text on your screen so we can see what EasyOCR actually read.
        import streamlit as st
        if raw_text:
            st.warning("🔍 DEBUG: EasyOCR read the following text:")
            st.code(raw_text, language="text")
        # ==========================================
        
        cleaned = self.clean_text(raw_text)
        if not cleaned:
            return []

        advisory_statements = []
        
        # Match "May contain..." or "Contains..."
        advisory_matches = re.finditer(r'\b(contains|may contain)\b[:\s]+(.*?)(?=\.|$)', cleaned, re.IGNORECASE)
        for match in advisory_matches:
            statement_content = match.group(2).lower()
            # Cleanly split advisory list on commas, semicolons, or 'and'
            items = re.split(r'[,;]|\band\b', statement_content)
            advisory_statements.extend([it.strip() for it in items if it.strip()])

        # Truncate main ingredient string before advisory statements start
        first_advisory = re.search(r'\b(contains|may contain)\b', cleaned, re.IGNORECASE)
        if first_advisory:
            cleaned = cleaned[:first_advisory.start()].strip()

        # Extract nested parenthetical ingredients
        sub_items = re.findall(r'\((.*?)\)', cleaned)
        sub_items_brackets = re.findall(r'\[(.*?)\]', cleaned)
        
        # Remove parenthetical text from main line
        main_text = re.sub(r'\(.*?\)', '', cleaned)
        main_text = re.sub(r'\[.*?\]', '', main_text)

        # Split main list on commas or semicolons
        raw_items = re.split(r'[,;]\s*', main_text)
        
        for sub in sub_items + sub_items_brackets:
            raw_items.extend(re.split(r'[,;]\s*', sub))

        # Include parsed advisory tokens as individual items
        raw_items.extend(advisory_statements)

        # Clean individual items
        parsed = []
        for item in raw_items:
            item_clean = re.sub(r'[^a-zA-Z0-9\s\-]', '', item).strip().lower()
            item_clean = re.sub(r'\b(ingredients|contains|may)\b', '', item_clean).strip()
            
            if item_clean and len(item_clean) > 2 and item_clean not in parsed:
                parsed.append(item_clean)

        return parsed

    def check_allergens(self, ingredients: list, user_allergies: list) -> list:
        """Matches extracted ingredients accurately against user selected allergies."""
        results = []
        # Normalize user allergies to lowercase
        allergies_lower = [a.lower().strip() for a in user_allergies]

        for item in ingredients:
            # Default state
            status = "safe"
            matched_group = None
            risk_level = None

            # 1. Check for exact or partial ingredient matches against the user's allergies
            for allergen in allergies_lower:
                # Create a generic base word (removes 's' for plural)
                if allergen.endswith('s') and len(allergen) > 3:
                    base_allergen = allergen[:-1]
                else:
                    base_allergen = allergen

                # Use pattern to find the allergen as a standalone word inside the ingredient text
                pattern = r'\b' + re.escape(base_allergen) + r's?\b'
                
                # ALLOW "may contain" to be caught
                if re.search(pattern, item, re.IGNORECASE):
                    status = "danger"
                    matched_group = allergen.capitalize()
                    risk_level = "High"
                    break

            results.append({
                "ingredient": item,
                "status": status,
                "allergen_group": matched_group,
                "risk_level": risk_level
            })

        return results

    def get_safety_summary(self, allergen_results: list) -> dict:
        danger_count = sum(1 for r in allergen_results if r.get('status') == 'danger')
        warning_count = sum(1 for r in allergen_results if r.get('status') == 'warning')
        safe_count = sum(1 for r in allergen_results if r.get('status') == 'safe')

        return {
            "is_safe": danger_count == 0,
            "danger": danger_count,
            "warning": warning_count,
            "safe": safe_count,
            "total": len(allergen_results)
        }