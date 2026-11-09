import requests
import pandas as pd
import time
import os


# Load cleaned dataset
df = pd.read_csv("data/clean_food_dataset.csv")

# Test only 5 products first
test_products = df.head(5)

results = []

headers = {
    "User-Agent": "IngredientCheckAI/1.0 student-project"
}

for _, row in test_products.iterrows():

    product_name = row["product_name"]

    url = "https://world.openfoodfacts.org/api/v2/search"

    params = {
        "search_terms": product_name,
        "page_size": 1,
        "fields": "code,product_name,brands,ingredients_text,allergens,nutriments"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        print(product_name, "→", response.status_code)

        if response.status_code == 200:

            data = response.json()
            products = data.get("products", [])

            if products:
                p = products[0]

                results.append({
                    "product_name": product_name,
                    "api_product_name": p.get("product_name"),
                    "api_brands": p.get("brands"),
                    "api_ingredients": p.get("ingredients_text"),
                    "api_allergens": p.get("allergens"),
                    "api_code": p.get("code")
                })

        time.sleep(7)

    except Exception as e:
        print("Error:", e)

api_df = pd.DataFrame(results)

os.makedirs("data/api_data", exist_ok=True)

api_df.to_csv(
    "data/api_data/openfoodfacts_data.csv",
    index=False
)

print("OpenFoodFacts data saved successfully!")

print("\nAPI TEST COMPLETE")
print(api_df)