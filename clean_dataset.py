import pandas as pd
import os

# ============================================================
# 1. INPUT DATASET
# ============================================================

possible_files = [
    "data/final_fused_with_fatsecret(1).csv",
    "data/final_fused_with_fatsecret.csv",
    "data/final_fused_food_dataset.csv"
]

input_file = None

for file in possible_files:
    if os.path.exists(file):
        input_file = file
        break

if input_file is None:
    raise FileNotFoundError(
        "Final fused dataset was not found inside the data folder."
    )

print("=" * 60)
print("LOADING FINAL FUSED DATASET")
print("=" * 60)

df = pd.read_csv(input_file, low_memory=False)

print("Input file:", input_file)
print("Original rows:", len(df))
print("Original columns:", len(df.columns))


# ============================================================
# 2. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df.dropna(how="all").reset_index(drop=True)


# ============================================================
# 3. REPLACE EMPTY STRINGS WITH NaN
# ============================================================

df = df.replace(r"^\s*$", pd.NA, regex=True)


# ============================================================
# 4. COLUMNS THAT ARE IDENTIFIERS
#    These should NOT receive a median value.
# ============================================================

id_columns = [
    "product_key",
    "edamam_food_id",
    "api_code",
    "cor_product_id",
    "usda_fdc_id",
    "fatsecret_food_id"
]

# Only use ID columns that actually exist
id_columns = [col for col in id_columns if col in df.columns]


# ============================================================
# 5. CONVERT IDENTIFIER COLUMNS TO TEXT
# ============================================================

for col in id_columns:
    df[col] = df[col].astype("string")


# ============================================================
# 6. NUMERIC COLUMNS
#    Missing numerical values are replaced by the median.
# ============================================================

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

# Don't treat identifiers as numerical measurements
numeric_columns = [
    col for col in numeric_columns
    if col not in id_columns
]

for col in numeric_columns:

    # Convert safely to numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate median
    median_value = df[col].median()

    # If a column is completely empty, use 0
    if pd.isna(median_value):
        median_value = 0

    df[col] = df[col].fillna(median_value)


# ============================================================
# 7. ALL OTHER COLUMNS
#    Missing text/category/boolean/API values -> "Unknown"
# ============================================================

remaining_columns = [
    col for col in df.columns
    if col not in numeric_columns
]

for col in remaining_columns:

    df[col] = df[col].astype("string")
    df[col] = df[col].fillna("Unknown")


# ============================================================
# 8. FINAL SAFETY CHECK
# ============================================================

# Replace any remaining missing values
# (extra protection against unusual pandas data types)
df = df.fillna("Unknown")


# ============================================================
# 9. CHECK DATASET SIZE
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET CHECK")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 10. CHECK NULL VALUES
# ============================================================

total_nulls = int(df.isna().sum().sum())

print("Total missing values:", total_nulls)


# ============================================================
# 11. CHECK COLUMN COUNT
# ============================================================

if len(df.columns) != 66:
    print(
        "\nWARNING: Expected 66 columns, "
        f"but found {len(df.columns)} columns."
    )
else:
    print("Column count check: PASSED")


# ============================================================
# 12. CHECK ROW COUNT
# ============================================================

if len(df) != 4959:
    print(
        "\nWARNING: Expected 4,959 rows, "
        f"but found {len(df)} rows."
    )
else:
    print("Row count check: PASSED")


# ============================================================
# 13. CHECK NULL STATUS
# ============================================================

if total_nulls == 0:
    print("Null value check: PASSED")
else:
    print("Null value check: FAILED")


# ============================================================
# 14. SAVE CLEAN DATASET
# ============================================================

output_file = "data/final_clean_dataset.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("CLEAN DATASET SAVED")
print("=" * 60)

print("File:", output_file)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))

print("\nCleaning completed successfully.")