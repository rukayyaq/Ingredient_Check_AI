import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from difflib import SequenceMatcher


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

APP_ID = os.getenv("EDAMAM_APP_ID")
APP_KEY = os.getenv("EDAMAM_APP_KEY")


if not APP_ID or not APP_KEY:
    print("ERROR: Edamam credentials not found.")
    print()
    print("Check your .env file:")
    print("EDAMAM_APP_ID=your_app_id")
    print("EDAMAM_APP_KEY=your_api_key")
    raise SystemExit


# =========================================================
# 2. LOAD DATASET
# =========================================================

DATASET_PATH = "data/clean_food_dataset.csv"

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    print("ERROR: Dataset not found.")
    print("Expected file:")
    print(DATASET_PATH)
    raise SystemExit


# =========================================================
# 3. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "product_name",
    "brands",
    "product_key"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("ERROR: Missing columns:")
    print(missing_columns)

    print()
    print("Available columns:")
    print(df.columns.tolist())

    raise SystemExit


# =========================================================
# 4. SELECT PRODUCTS
# =========================================================

# Start with 50 products
products = df[
    required_columns
].head(50).copy()

print()
print("=" * 50)
print("EDAMAM API DATA COLLECTION")
print("=" * 50)

print("Main dataset rows:", len(df))
print("Products being tested:", len(products))


# =========================================================
# 5. API URL
# =========================================================

URL = (
    "https://api.edamam.com/"
    "api/food-database/v2/parser"
)


# =========================================================
# 6. RESULT STORAGE
# =========================================================

results = []


# =========================================================
# 7. SIMILARITY FUNCTION
# =========================================================

def similarity(text1, text2):

    text1 = str(text1).lower().strip()
    text2 = str(text2).lower().strip()

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# =========================================================
# 8. PROCESS PRODUCTS
# =========================================================

for count, (_, row) in enumerate(
    products.iterrows(),
    start=1
):

    product_name = str(
        row["product_name"]
    ).strip()

    brand = str(
        row["brands"]
    ).strip()

    product_key = str(
        row["product_key"]
    ).strip()


    # -----------------------------------------------------
    # SEARCH TEXT
    # -----------------------------------------------------

    search_text = f"{product_name} {brand}"


    # -----------------------------------------------------
    # API PARAMETERS
    # -----------------------------------------------------

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "ingr": search_text
    }


    print(
        f"{count}/{len(products)} "
        f"{product_name}",
        end=" -> "
    )


    try:

        response = requests.get(
            URL,
            params=params,
            timeout=20
        )


        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            print("200")

            data = response.json()

            hints = data.get(
                "hints",
                []
            )


            if not hints:

                print(
                    "   No API result found."
                )

                time.sleep(1)

                continue


            # -------------------------------------------------
            # FIND BEST API RESULT
            # -------------------------------------------------

            best_food = None
            best_score = 0.0


            for hint in hints:

                food = hint.get(
                    "food",
                    {}
                )

                api_name = food.get(
                    "label",
                    ""
                )


                score = similarity(
                    product_name,
                    api_name
                )


                if score > best_score:

                    best_score = score
                    best_food = food


            # -------------------------------------------------
            # SAVE BEST RESULT
            # -------------------------------------------------

            if best_food:

                nutrients = best_food.get(
                    "nutrients",
                    {}
                )


                result = {

                    "product_key":
                        product_key,

                    "original_product":
                        product_name,

                    "original_brand":
                        brand,

                    "edamam_food_id":
                        best_food.get(
                            "foodId"
                        ),

                    "edamam_product_name":
                        best_food.get(
                            "label"
                        ),

                    "edamam_brand":
                        best_food.get(
                            "brand"
                        ),

                    "edamam_category":
                        best_food.get(
                            "category"
                        ),

                    "api_calories":
                        nutrients.get(
                            "ENERC_KCAL"
                        ),

                    "api_protein":
                        nutrients.get(
                            "PROCNT"
                        ),

                    "api_fat":
                        nutrients.get(
                            "FAT"
                        ),

                    "api_carbs":
                        nutrients.get(
                            "CHOCDF"
                        ),

                    "api_fiber":
                        nutrients.get(
                            "FIBTG"
                        ),

                    "api_ingredients":
                        best_food.get(
                            "foodContentsLabel"
                        ),

                    "match_score":
                        round(
                            best_score,
                            3
                        )
                }


                results.append(result)


                print(
                    f"   Best match: "
                    f"{best_food.get('label', 'Unknown')}"
                )

                print(
                    f"   Score: "
                    f"{round(best_score, 3)}"
                )


        # =================================================
        # RATE LIMIT
        # =================================================

        elif response.status_code == 429:

            print("429 - Rate limit")

            print(
                "   Waiting 10 seconds..."
            )

            time.sleep(10)


        # =================================================
        # OTHER API ERROR
        # =================================================

        else:

            print(
                f"{response.status_code}"
            )

            print(
                "   API request failed."
            )


    # =====================================================
    # NETWORK ERROR
    # =====================================================

    except requests.exceptions.Timeout:

        print(
            "TIMEOUT"
        )

        print(
            "   Request took too long."
        )


    except requests.exceptions.ConnectionError:

        print(
            "CONNECTION ERROR"
        )

        print(
            "   Check internet connection."
        )


    except requests.exceptions.RequestException as error:

        print(
            "REQUEST ERROR"
        )

        print(
            "   ", error
        )


    except Exception as error:

        print(
            "ERROR"
        )

        print(
            "   ", error
        )


    # -----------------------------------------------------
    # DELAY BETWEEN REQUESTS
    # -----------------------------------------------------

    time.sleep(1)


# =========================================================
# 9. CREATE DATAFRAME
# =========================================================

api_df = pd.DataFrame(
    results
)


# =========================================================
# 10. CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = "data/api_data"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# 11. REMOVE DUPLICATE PRODUCT KEYS
# =========================================================

if not api_df.empty:

    duplicate_count = api_df[
        "product_key"
    ].duplicated().sum()

    print()
    print(
        "Duplicate API keys:",
        duplicate_count
    )


    api_df = api_df.drop_duplicates(
        subset=["product_key"],
        keep="first"
    )


# =========================================================
# 12. SAVE CSV
# =========================================================

OUTPUT_FILE = (
    "data/api_data/edamam_data.csv"
)


api_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 13. FINAL REPORT
# =========================================================

print()
print("=" * 50)
print("EDAMAM DATA COLLECTION COMPLETE")
print("=" * 50)

print(
    "Main products tested:",
    len(products)
)

print(
    "API records collected:",
    len(api_df)
)

print(
    "API columns:",
    len(api_df.columns)
)

print(
    "Saved to:",
    OUTPUT_FILE
)


# =========================================================
# 14. SHOW RESULTS
# =========================================================

if not api_df.empty:

    print()
    print("API RESULTS")
    print("-" * 50)

    columns_to_show = [
        "original_product",
        "edamam_product_name",
        "match_score"
    ]

    print(
        api_df[
            columns_to_show
        ].to_string(index=False)
    )

else:

    print()
    print("WARNING:")
    print("No API records were collected.")


print()
print("DONE!")