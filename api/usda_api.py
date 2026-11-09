import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from difflib import SequenceMatcher

# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv()

API_KEY = os.getenv("USDA_API_KEY")

if not API_KEY:
    print("ERROR: USDA_API_KEY not found in .env")
    raise SystemExit


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/clean_food_dataset.csv"
)

products = df[
    ["product_name", "brands", "product_key"]
].head(20).copy()

print("=" * 50)
print("USDA FOODDATA CENTRAL API")
print("=" * 50)
print("Products to test:", len(products))


# ==========================================
# 3. API
# ==========================================

URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

results = []


# ==========================================
# 4. SIMILARITY
# ==========================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        str(a).lower().strip(),
        str(b).lower().strip()
    ).ratio()


# ==========================================
# 5. SEARCH PRODUCTS
# ==========================================

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

    search_text = f"{product_name} {brand}"

    params = {
        "api_key": API_KEY,
        "query": search_text,
        "pageSize": 5
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

        print(response.status_code)

        # ======================================
        # SUCCESS
        # ======================================

        if response.status_code == 200:

            data = response.json()

            foods = data.get(
                "foods",
                []
            )

            if not foods:

                print(
                    "   No USDA result found."
                )

                time.sleep(1)
                continue


            # ==================================
            # FIND BEST MATCH
            # ==================================

            best_food = None
            best_score = 0.0

            for food in foods:

                api_name = food.get(
                    "description",
                    ""
                )

                score = similarity(
                    product_name,
                    api_name
                )

                if score > best_score:

                    best_score = score
                    best_food = food


            # ==================================
            # SAVE RESULT
            # ==================================

            if best_food:

                results.append({

                    "product_key":
                        product_key,

                    "original_product":
                        product_name,

                    "original_brand":
                        brand,

                    "usda_fdc_id":
                        best_food.get(
                            "fdcId"
                        ),

                    "usda_product_name":
                        best_food.get(
                            "description"
                        ),

                    "usda_brand":
                        best_food.get(
                            "brandOwner"
                        ),

                    "usda_data_type":
                        best_food.get(
                            "dataType"
                        ),

                    "usda_match_score":
                        round(
                            best_score,
                            3
                        ),

                    "usda_calories":
                        best_food.get(
                            "foodNutrients",
                            [{}]
                        )[0].get(
                            "value"
                        )
                })

                print(
                    "   Match:",
                    best_food.get(
                        "description"
                    )
                )

                print(
                    "   Score:",
                    round(
                        best_score,
                        3
                    )
                )


        # ======================================
        # RATE LIMIT
        # ======================================

        elif response.status_code == 429:

            print(
                "   Rate limit reached."
            )

            time.sleep(5)


        # ======================================
        # INVALID KEY
        # ======================================

        elif response.status_code in [401, 403]:

            print(
                "   ERROR: Invalid USDA API key."
            )

            break


        # ======================================
        # OTHER ERROR
        # ======================================

        else:

            print(
                "   API error:",
                response.text[:200]
            )


    except requests.exceptions.Timeout:

        print(
            "   Request timed out."
        )


    except requests.exceptions.RequestException as error:

        print(
            "   Request error:",
            error
        )


    except Exception as error:

        print(
            "   Unexpected error:",
            error
        )


    time.sleep(0.5)


# ==========================================
# 6. CREATE DATAFRAME
# ==========================================

usda_df = pd.DataFrame(
    results
)


# ==========================================
# 7. REMOVE DUPLICATES
# ==========================================

if not usda_df.empty:

    usda_df = usda_df.drop_duplicates(
        subset=["product_key"],
        keep="first"
    )


# ==========================================
# 8. SAVE
# ==========================================

os.makedirs(
    "data/api_data",
    exist_ok=True
)

output_file = (
    "data/api_data/usda_data.csv"
)

usda_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 9. FINAL REPORT
# ==========================================

print()
print("=" * 50)
print("USDA API COMPLETE")
print("=" * 50)

print(
    "Products tested:",
    len(products)
)

print(
    "USDA records:",
    len(usda_df)
)

print(
    "USDA columns:",
    len(usda_df.columns)
)

print(
    "Saved:",
    output_file
)

if not usda_df.empty:

    print()
    print(
        usda_df[
            [
                "original_product",
                "usda_product_name",
                "usda_brand",
                "usda_match_score"
            ]
        ].to_string(index=False)
    )

else:

    print("No USDA records collected.")

print()
print("DONE!")