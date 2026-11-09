import os
import re
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from difflib import SequenceMatcher


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

API_DATA_DIR = os.path.join(
    DATA_DIR,
    "api_data"
)

os.makedirs(API_DATA_DIR, exist_ok=True)


# ============================================================
# LOAD .ENV
# ============================================================

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_FILE)

CLIENT_ID = os.getenv(
    "FATSECRET_CLIENT_ID"
)

CLIENT_SECRET = os.getenv(
    "FATSECRET_CLIENT_SECRET"
)


if not CLIENT_ID or not CLIENT_SECRET:

    raise ValueError(
        "FATSECRET_CLIENT_ID or FATSECRET_CLIENT_SECRET "
        "is missing from .env"
    )


# ============================================================
# FATSECRET API URL
# ============================================================

TOKEN_URL = (
    "https://oauth.fatsecret.com/connect/token"
)

API_URL = (
    "https://platform.fatsecret.com/rest/server.api"
)


# ============================================================
# GET ACCESS TOKEN
# ============================================================

def get_access_token():

    print()
    print("=" * 60)
    print("GETTING FATSECRET ACCESS TOKEN")
    print("=" * 60)

    response = requests.post(

        TOKEN_URL,

        data={
            "grant_type": "client_credentials",
            "scope": "basic"
        },

        auth=(
            CLIENT_ID,
            CLIENT_SECRET
        ),

        timeout=30
    )

    print(
        "Token status:",
        response.status_code
    )

    if response.status_code != 200:

        print(
            "Token error:"
        )

        print(
            response.text
        )

        return None

    try:

        data = response.json()

    except Exception:

        print(
            "Could not read token response."
        )

        print(
            response.text
        )

        return None

    access_token = data.get(
        "access_token"
    )

    if not access_token:

        print(
            "Access token missing."
        )

        print(data)

        return None

    print(
        "FatSecret access token obtained successfully."
    )

    return access_token


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if pd.isna(text):

        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIMILARITY SCORE
# ============================================================

def similarity_score(
    original,
    api_name
):

    original = normalize_text(
        original
    )

    api_name = normalize_text(
        api_name
    )

    if not original or not api_name:

        return 0.0

    return round(
        SequenceMatcher(
            None,
            original,
            api_name
        ).ratio(),
        3
    )


# ============================================================
# PARSE NUTRITION
# ============================================================

def parse_nutrition(
    description
):

    result = {

        "fatsecret_calories": None,

        "fatsecret_fat": None,

        "fatsecret_carbs": None,

        "fatsecret_protein": None
    }

    if not description:

        return result


    # Calories
    match = re.search(
        r"Calories:\s*([\d.]+)",
        description,
        re.IGNORECASE
    )

    if match:

        result[
            "fatsecret_calories"
        ] = float(
            match.group(1)
        )


    # Fat
    match = re.search(
        r"Fat:\s*([\d.]+)g",
        description,
        re.IGNORECASE
    )

    if match:

        result[
            "fatsecret_fat"
        ] = float(
            match.group(1)
        )


    # Carbs
    match = re.search(
        r"Carbs:\s*([\d.]+)g",
        description,
        re.IGNORECASE
    )

    if match:

        result[
            "fatsecret_carbs"
        ] = float(
            match.group(1)
        )


    # Protein
    match = re.search(
        r"Protein:\s*([\d.]+)g",
        description,
        re.IGNORECASE
    )

    if match:

        result[
            "fatsecret_protein"
        ] = float(
            match.group(1)
        )


    return result


# ============================================================
# SEARCH FATSECRET
# ============================================================

def search_fatsecret(
    product_name,
    access_token
):

    headers = {

        "Authorization":
            f"Bearer {access_token}",

        "Content-Type":
            "application/json"
    }


    # FatSecret official method-based API
    params = {

        "method":
            "foods.search",

        "search_expression":
            product_name,

        "page_number":
            0,

        "max_results":
            10,

        "format":
            "json"
    }


    try:

        response = requests.post(

            API_URL,

            headers=headers,

            params=params,

            timeout=30
        )

    except requests.RequestException as e:

        print(
            "   Request error:",
            e
        )

        return None


    print(
        f"{product_name} -> "
        f"{response.status_code}"
    )


    # ========================================================
    # STATUS CHECK
    # ========================================================

    if response.status_code != 200:

        print(
            "   ERROR RESPONSE:"
        )

        print(
            response.text[:1000]
        )

        return None


    # ========================================================
    # JSON
    # ========================================================

    try:

        data = response.json()

    except Exception:

        print(
            "   Invalid JSON response."
        )

        print(
            response.text[:1000]
        )

        return None


    # ========================================================
    # DEBUG RESPONSE
    # ========================================================

    print(
        "   Response received."
    )


    # ========================================================
    # CHECK FATSECRET ERROR
    # ========================================================

    if "error" in data:

        print(
            "   FATSECRET ERROR:"
        )

        print(
            data["error"]
        )

        return None


    # ========================================================
    # GET FOODS
    # ========================================================

    foods_container = data.get(
        "foods"
    )


    if not foods_container:

        print(
            "   No 'foods' object."
        )

        print(
            "   Response:"
        )

        print(
            str(data)[:1500]
        )

        return None


    # ========================================================
    # SHOW RESULT COUNT
    # ========================================================

    total_results = foods_container.get(
        "total_results",
        0
    )

    print(
        "   Total results:",
        total_results
    )


    # ========================================================
    # GET FOOD
    # ========================================================

    foods = foods_container.get(
        "food",
        []
    )


    # If only one result, FatSecret returns a dictionary
    if isinstance(
        foods,
        dict
    ):

        foods = [foods]


    if not isinstance(
        foods,
        list
    ):

        print(
            "   Unexpected food format."
        )

        print(
            str(foods)[:1000]
        )

        return None


    if len(foods) == 0:

        print(
            "   No FatSecret result."
        )

        return None


    # ========================================================
    # FIND BEST MATCH
    # ========================================================

    best_food = None

    best_score = 0.0


    for food in foods:

        food_name = food.get(
            "food_name",
            ""
        )

        score = similarity_score(

            product_name,

            food_name
        )


        if score > best_score:

            best_score = score

            best_food = food


    if best_food is None:

        print(
            "   No usable result."
        )

        return None


    # ========================================================
    # EXTRACT DATA
    # ========================================================

    food_id = best_food.get(
        "food_id"
    )

    food_name = best_food.get(
        "food_name"
    )

    brand_name = best_food.get(
        "brand_name"
    )

    food_type = best_food.get(
        "food_type"
    )

    description = best_food.get(
        "food_description"
    )


    # ========================================================
    # NUTRITION
    # ========================================================

    nutrition = parse_nutrition(
        description
    )


    print(
        "   Match:",
        food_name
    )

    print(
        "   Score:",
        best_score
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "fatsecret_food_id":
            food_id,

        "fatsecret_product_name":
            food_name,

        "fatsecret_brand":
            brand_name,

        "fatsecret_food_type":
            food_type,

        "fatsecret_description":
            description,

        "fatsecret_match_score":
            best_score,

        "fatsecret_calories":
            nutrition[
                "fatsecret_calories"
            ],

        "fatsecret_fat":
            nutrition[
                "fatsecret_fat"
            ],

        "fatsecret_carbs":
            nutrition[
                "fatsecret_carbs"
            ],

        "fatsecret_protein":
            nutrition[
                "fatsecret_protein"
            ]
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print("=" * 60)
print("FATSECRET API TEST")
print("=" * 60)


# ============================================================
# GET TOKEN
# ============================================================

access_token = get_access_token()


if not access_token:

    print(
        "Could not obtain access token."
    )

    raise SystemExit


# ============================================================
# LOAD FINAL DATASET
# ============================================================

input_file = os.path.join(
    DATA_DIR,
    "final_fused_food_dataset.csv"
)


print()
print(
    "Dataset path:"
)

print(
    input_file
)


if not os.path.exists(
    input_file
):

    print()
    print(
        "ERROR: Dataset file not found!"
    )

    print(
        "Expected file:"
    )

    print(
        input_file
    )

    raise SystemExit


df = pd.read_csv(
    input_file
)


print()
print(
    "Dataset rows:",
    len(df)
)


# ============================================================
# CHECK COLUMNS
# ============================================================

print(
    "Dataset columns:",
    len(df.columns)
)


if "product_name" not in df.columns:

    raise ValueError(
        "product_name column does not exist."
    )


if "brands" not in df.columns:

    print(
        "WARNING: brands column not found."
    )

    df["brands"] = ""


# ============================================================
# FIRST 20 PRODUCTS
# ============================================================

test_df = df[
    [
        "product_name",
        "brands"
    ]
].head(20).copy()


print(
    "Products to test:",
    len(test_df)
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TEST PRODUCTS
# ============================================================

for position, (index, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    product_name = row[
        "product_name"
    ]


    if pd.isna(
        product_name
    ):

        continue


    product_name = str(
        product_name
    ).strip()


    if not product_name:

        continue


    print()
    print(
        f"{position}/"
        f"{len(test_df)} "
        f"{product_name}"
    )


    result = search_fatsecret(

        product_name,

        access_token
    )


    if result:

        result[
            "original_product"
        ] = product_name


        result[
            "original_brand"
        ] = row[
            "brands"
        ]


        result[
            "product_key"
        ] = normalize_text(
            product_name
        )


        results.append(
            result
        )


    # Avoid sending requests too quickly
    time.sleep(
        0.5
    )


# ============================================================
# DATAFRAME
# ============================================================

fatsecret_df = pd.DataFrame(
    results
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("FATSECRET API COMPLETE")
print("=" * 60)


print(
    "Products tested:",
    len(test_df)
)


print(
    "FatSecret records:",
    len(fatsecret_df)
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    API_DATA_DIR,
    "fatsecret_data.csv"
)


if len(fatsecret_df) > 0:

    fatsecret_df.to_csv(

        output_file,

        index=False
    )


    print(
        "FatSecret columns:",
        len(
            fatsecret_df.columns
        )
    )


    print(
        "Saved:",
        output_file
    )


    print()
    print(
        "RESULT PREVIEW:"
    )


    preview_columns = [

        "original_product",

        "original_brand",

        "fatsecret_product_name",

        "fatsecret_brand",

        "fatsecret_match_score",

        "fatsecret_calories",

        "fatsecret_protein",

        "fatsecret_fat",

        "fatsecret_carbs"
    ]


    existing_columns = [

        col

        for col in preview_columns

        if col in fatsecret_df.columns
    ]


    print(
        fatsecret_df[
            existing_columns
        ]
    )


else:

    print()
    print(
        "No FatSecret records collected."
    )

    print(
        "fatsecret_data.csv was not populated."
    )


print()
print("DONE!")