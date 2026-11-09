import pandas as pd
import numpy as np
from pathlib import Path

# Load both datasets
base_dir = Path(__file__).resolve().parent
existing_df = pd.read_csv(base_dir / "data" / "final_clean_dataset.csv")
new_df = pd.read_csv(base_dir / "data" / "packaged_foods_india.csv")

print(f"Existing dataset: {existing_df.shape}")
print(f"New dataset: {new_df.shape}")

# Rename columns in new dataset to match existing
new_df = new_df.rename(columns={
    'Item name': 'product_name',
    'Brand_Name': 'brands',
    'Category': 'categories',
    'Sub_Category': 'sub_categories',
    'Ingredients': 'ingredients_text',
    'Calories_kcal': 'energy_kcal',
    'Carbohydrates_g': 'carbohydrates_100g',
    'Proteins_g': 'proteins_100g',
    'Total_Fat_g': 'fat_100g',
    'Saturated_Fat_g': 'saturated_fat_100g',
    'Trans_Fat_g': 'trans_fat_100g',
    'Sugar_g': 'sugars_100g',
    'Sodium_mg': 'sodium_mg',
    'Dietary_Fiber_g': 'fiber_100g',
    'Cholesterol_mg': 'cholesterol_mg',
    'Calcium_mg': 'calcium_mg',
    'Iron_mg': 'iron_mg',
    'Price_INR': 'price_inr',
    'Serving_Size_g': 'serving_size_g'
})

# Add source column
existing_df['data_source'] = 'Phase1_Fused'
new_df['data_source'] = 'Packaged_Foods_India'

# Concatenate datasets
fused_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

# **FIX: Use `str` dtype, NOT `object` for pandas 2.x**
# Fill categorical NaN with 'Unknown' - but FIRST convert to string
str_cols = fused_df.select_dtypes(include=['str', 'object']).columns
for col in str_cols:
    if col in fused_df.columns:
        fused_df[col] = fused_df[col].fillna('Unknown')

# Fill numeric NaN with median
numeric_cols = fused_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in fused_df.columns:
        fused_df[col] = fused_df[col].fillna(fused_df[col].median())

# Save
output_path = base_dir / "data" / "new_fused_dataset.csv"
try:
    fused_df.to_csv(output_path, index=False)
    print(f"\n✅ New fused dataset saved: {fused_df.shape}")
    print(f"Total records: {len(fused_df)}")
    print(f"Total columns: {len(fused_df.columns)}")
except PermissionError:
    print("\n❌ ERROR: Cannot save file. The file is in use.")
    print("Please close the file in Excel/Streamlit/VS Code, then run again.")