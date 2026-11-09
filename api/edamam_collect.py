import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("EDAMAM_APP_ID")
APP_KEY = os.getenv("EDAMAM_APP_KEY")

# Load our cleaned dataset
df = pd.read_csv("data/clean_food_dataset.csv")

# Start with only 50 products
products = df[["product_name", "brands", "product_key"]].head(50)

results = []

url = "https://api.edamam.com/api/food-database/v2/parser"

for i, row in products.iterrows():

    food_name = row["product_name"]

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "ingr": food_name
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        print(i + 1, "/", len(products),
              food_name,
              "->", response.status_code)

        if response.status_code == 200:

            data = response.json()
            hints = data.get("hints", [])

            if hints:

                food = hints[0]["food"]

                results.append({
                    "product_key": row["product_key"],
                    "original_product": food_name,
                    "edamam_food_id": food.get("foodId"),
                    "edamam_product_name": food.get("label"),
                    "edamam_brand": food.get("brand"),
                    "edamam_category": food.get("category"),

                    "api_calories": food.get(
                        "nutrients", {}
                    ).get("ENERC_KCAL"),

                    "api_protein": food.get(
                        "nutrients", {}
                    ).get("PROCNT"),

                    "api_fat": food.get(
                        "nutrients", {}
                    ).get("FAT"),

                    "api_carbs": food.get(
                        "nutrients", {}
                    ).get("CHOCDF"),

                    "api_fiber": food.get(
                        "nutrients", {}
                    ).get("FIBTG"),

                    "api_ingredients": food.get(
                        "foodContentsLabel"
                    )
                })

        time.sleep(1)

    except Exception as e:
        print("Error:", e)


api_df = pd.DataFrame(results)

os.makedirs("data/api_data", exist_ok=True)

api_df.to_csv(
    "data/api_data/edamam_data.csv",
    index=False
)

print("\n========================")
print("EDAMAM DATA CREATED")
print("========================")
print("API rows:", len(api_df))
print("API columns:", len(api_df.columns))
print(api_df.head())