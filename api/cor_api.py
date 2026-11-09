import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv()

API_KEY = os.getenv("COR_API_KEY")

if not API_KEY:
    print("ERROR: COR_API_KEY not found in .env")
    raise SystemExit


# ==========================================
# 2. LOAD CLEAN DATASET
# ==========================================

df = pd.read_csv(
    "data/clean_food_dataset.csv"
)

products = df[
    ["product_name", "brands", "product_key"]
].head(20).copy()

print("=" * 50)
print("c0r.ai FOOD API TEST")
print("=" * 50)
print("Products:", len(products))


# ==========================================
# 3. API SETTINGS
# ==========================================

URL = "https://api.c0r.ai/v1/search"

headers = {
    "X-API-Key": API_KEY
}

results = []


# ==========================================
# 4. SEARCH PRODUCTS
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

    # Use product name only first
    search_text = product_name

    params = {
        "q": search_text,
        "lang": "en",
        "limit": 5
    }

    print(
        f"{count}/{len(products)} "
        f"{product_name}",
        end=" -> "
    )

    try:

        response = requests.get(
            URL,
            headers=headers,
            params=params,
            timeout=20
        )

        print(response.status_code)

        # ======================================
        # SUCCESS
        # ======================================

        if response.status_code == 200:

            data = response.json()

            # c0r.ai returns search results
            items = data.get(
                "results",
                data.get("data", [])
            )

            if items:

                best = items[0]

                results.append({

                    "product_key":
                        product_key,

                    "original_product":
                        product_name,

                    "original_brand":
                        brand,

                    "cor_product_id":
                        best.get("id"),

                    "cor_product_name":
                        best.get(
                            "name",
                            best.get("label")
                        ),

                    "cor_brand":
                        best.get("brand"),

                    "cor_calories":
                        best.get(
                            "calories"
                        ),

                    "cor_protein":
                        best.get(
                            "protein"
                        ),

                    "cor_fat":
                        best.get(
                            "fat"
                        ),

                    "cor_carbs":
                        best.get(
                            "carbs"
                        )
                })

                print(
                    "   Match:",
                    best.get(
                        "name",
                        best.get("label", "Unknown")
                    )
                )

            else:

                print(
                    "   No result found."
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
        # UNAUTHORIZED
        # ======================================

        elif response.status_code == 401:

            print(
                "   ERROR: Invalid API key."
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
            "   Timeout."
        )


    except requests.exceptions.RequestException as e:

        print(
            "   Request error:",
            e
        )


    except Exception as e:

        print(
            "   Error:",
            e
        )


    time.sleep(1)


# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

api_df = pd.DataFrame(results)


# ==========================================
# 6. SAVE DATA
# ==========================================

os.makedirs(
    "data/api_data",
    exist_ok=True
)

output_file = (
    "data/api_data/cor_data.csv"
)

api_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 7. FINAL REPORT
# ==========================================

print()
print("=" * 50)
print("c0r.ai API COMPLETE")
print("=" * 50)

print(
    "Products tested:",
    len(products)
)

print(
    "API records:",
    len(api_df)
)

print(
    "Columns:",
    len(api_df.columns)
)

print(
    "Saved:",
    output_file
)


if not api_df.empty:

    print()
    print(
        api_df.to_string(
            index=False
        )
    )

else:

    print(
        "No records were returned."
    )