import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load fused dataset
base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "data" / "new_fused_dataset.csv")

print(f"Fused dataset shape: {df.shape}")

# ============================================================
# CHECK COLUMN NAMES (For Debugging)
# ============================================================
print("\nColumns in dataset:")
for col in df.columns:
    print(f"  - {col}")

# ============================================================
# USE THE CORRECT COLUMN NAMES (From fusion_script.py)
# ============================================================

# These are the RENAMED columns from fusion_script.py
num_cols = ['energy_kcal', 'fat_100g', 'sugars_100g', 'proteins_100g', 
            'carbohydrates_100g', 'sodium_mg', 'price_inr', 'serving_size_g']

# Check which columns exist
existing_cols = [c for c in num_cols if c in df.columns]
print(f"\nNumeric columns found: {existing_cols}")

# Convert to numeric (if not already)
for col in existing_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# 1. UNIVARIATE ANALYSIS
# ============================================================
print("\n=== UNIVARIATE ANALYSIS ===")

# Distribution of Price (price_inr)
plt.figure(figsize=(10, 5))
sns.histplot(df['price_inr'].dropna(), bins=30, kde=True)
plt.title('Distribution of Price (INR)')
plt.xlabel('Price (INR)')
plt.ylabel('Count')
plt.savefig('eda_price_distribution.png')
plt.show()
plt.close()

# Box plot for Serving Size (serving_size_g)
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['serving_size_g'].dropna())
plt.title('Serving Size Distribution (g)')
plt.ylabel('Serving Size (g)')
plt.savefig('eda_serving_size.png')
plt.show()
plt.close()

# ============================================================
# 2. BIVARIATE ANALYSIS
# ============================================================
print("\n=== BIVARIATE ANALYSIS ===")

# Calories vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='energy_kcal', y='price_inr', data=df, alpha=0.6)
plt.title('Calories vs Price (INR)')
plt.xlabel('Calories (kcal)')
plt.ylabel('Price (INR)')
plt.savefig('eda_calories_vs_price.png')
plt.show()
plt.close()

# Calories by Data Source
if 'data_source' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='data_source', y='energy_kcal', data=df)
    plt.title('Calories Distribution by Data Source')
    plt.xlabel('Data Source')
    plt.ylabel('Calories (kcal)')
    plt.savefig('eda_calories_by_source.png')
    plt.show()
    plt.close()
else:
    print("'data_source' column not found - skipping this plot")

# ============================================================
# 3. MULTIVARIATE ANALYSIS
# ============================================================
print("\n=== MULTIVARIATE ANALYSIS ===")

# Create clean dataframe for correlation
corr_df = df[existing_cols].dropna()
print(f"Rows used for correlation: {len(corr_df)}")

# Correlation matrix
corr = corr_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap - Fused Dataset')
plt.savefig('eda_fused_correlation.png')
plt.show()
plt.close()

# ============================================================
# 4. COMPARISON BETWEEN DATASETS
# ============================================================
print("\n=== COMPARISON: PHASE 1 vs PACKAGED FOODS INDIA ===")

if 'data_source' in df.columns:
    # Group by data_source and calculate means
    nutrition_cols = ['energy_kcal', 'fat_100g', 'sugars_100g', 'proteins_100g']
    nutrition_cols = [c for c in nutrition_cols if c in df.columns]

    comparison = df.groupby('data_source')[nutrition_cols].mean().round(2)
    print("\nAverage Nutrition by Data Source:")
    print(comparison)

    # Bar plot comparison
    plt.figure(figsize=(12, 6))
    comparison.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Nutrition by Data Source')
    plt.xlabel('Data Source')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('eda_comparison.png')
    plt.show()
    plt.close()
else:
    print("'data_source' column not found - skipping comparison")

# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================
print("\n=== SUMMARY STATISTICS ===")
print("\nPrice (INR):")
print(df['price_inr'].describe())

print("\nServing Size (g):")
print(df['serving_size_g'].describe())

print("\nEnergy (kcal):")
print(df['energy_kcal'].describe())

print("\nFat (g):")
print(df['fat_100g'].describe())

print("\n✅ EDA on fused dataset complete!")
print("\nPNG files saved:")
print("  - eda_price_distribution.png")
print("  - eda_serving_size.png")
print("  - eda_calories_vs_price.png")
print("  - eda_calories_by_source.png")
print("  - eda_fused_correlation.png")
print("  - eda_comparison.png")