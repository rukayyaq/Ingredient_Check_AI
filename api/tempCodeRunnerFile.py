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

API_KEY = os.getenv("SPOONACULAR_API_KEY")

if not API_KEY:
    print("ERROR: SPOONACULAR_API_KEY not found in .env")
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

print("=" * 55)
print("SPOONACULAR API TEST")
print("=" * 55)
print("Products to test:", len(products))


# ==========================================
# 3. API ENDPOINT
# ==========================================

URL = "https://api.spoonacular.com/food/products/search"

results = []


# ==========================================
# 4. SIMILARITY FUNCTION
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
        "apiKey": API_KEY,
        "query": search_text,
        "number": 5
    }

    print(
        f"{count}/{len(products)} "
        f"{product_name} -> ",
        end=""
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

            products_found = data.get(
                "products",
                []
            )

            if not products_found:

                print(
                    "   No Spoonacular result found."
                )

                time.sleep(1)
                continue


            # ==================================
            # FIND BEST MATCH
            # ==================================

            best_product = None
            best_score = 0.0

            for product in products_found:

                api_name = product.get(
                    "title",
                    ""
                )

                score = similarity(
                    product_name,
                    api_name
                )

                if score > best_score:

                    best_score = score
                    best_product = product


            # ==================================
            # STORE RESULT
            # ==================================

            if best_product:

                results.append({

                    "product_key":
                        product_key,

                    "original_product":
                        product_name,

                    "original_brand":
                        brand,

                    "spoon_product_id":
                        best_product.get(
                            "id"
                        ),

                    "spoon_product_name":
                        best_product.get(
                            "title"
                        ),

                    "spoon_brand":
                        best_product.get(
                            "brand"
                        ),

                    "spoon_match_score":
                        round(
                            best_score,
                            3
                        )
                })

                print(
                    "   Match:",
                    best_product.get(
                        "title"
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

        elif response.status_code == 402:

            print(
                "   API quota/payment limit reached."
            )

            break


        # ======================================
        # INVALID KEY
        # ======================================

        elif response.status_code in [401, 403]:

            print(
                "   ERROR: Invalid Spoonacular API key."
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


    time.sleep(1)


# ==========================================
# 6. CREATE DATAFRAME
# ==========================================

spoon_df = pd.DataFrame(
    results
)


# ==========================================
# 7. REMOVE DUPLICATES
# ==========================================

if not spoon_df.empty:

    spoon_df = spoon_df.drop_duplicates(
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
    "data/api_data/spoonacular_data.csv"
)

spoon_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 9. FINAL REPORT
# ==========================================

print()
print("=" * 55)
print("SPOONACULAR API COMPLETE")
print("=" * 55)

print(
    "Products tested:",
    len(products)
)

print(
    "Spoonacular records:",
    len(spoon_df)
)

print(
    "Spoonacular columns:",
    len(spoon_df.columns)
)

print(
    "Saved:",
    output_file
)

if not spoon_df.empty:

    print()

    print(
        spoon_df[
            [
                "original_product",
                "spoon_product_name",
                "spoon_brand",
                "spoon_match_score"
            ]
        ].to_string(index=False)
    )

else:

    print("No Spoonacular records collected.")

print()
print("DONE!")