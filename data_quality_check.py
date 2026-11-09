import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# LOAD FUSED DATASET
# ============================================================
base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "data" / "new_fused_dataset.csv")

print("=" * 60)
print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
print("=" * 60)

# ============================================================
# 1. STRUCTURAL CHECKS
# ============================================================
print("\n1. STRUCTURAL CHECKS")
print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Data Types:\n{df.dtypes.value_counts()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check unique values in key columns
print("\nUnique values (sample columns):")
print(f"  - product_name: {df['product_name'].nunique()} unique")
print(f"  - brands: {df['brands'].nunique()} unique")
print(f"  - categories: {df['categories'].nunique()} unique")

# ============================================================
# 2. MISSING DATA - NaN VALUES
# ============================================================
print("\n2. MISSING DATA - NaN (None) VALUES")
nan_counts = df.isnull().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
print(nan_counts)
print(f"Total NaN: {nan_counts.sum()}")

# ============================================================
# 3. MISSING DATA - 'UNKNOWN' VALUES
# ============================================================
print("\n3. MISSING DATA - 'UNKNOWN' VALUES (Imputed missing data)")
unknown_counts = {}
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'str':
        count = (df[col] == 'Unknown').sum()
        if count > 0:
            unknown_counts[col] = count

unknown_series = pd.Series(unknown_counts).sort_values(ascending=False)
print(unknown_series)
print(f"Total 'Unknown': {unknown_series.sum()}")

# ============================================================
# 4. TOTAL MISSING DATA (NaN + Unknown)
# ============================================================
print("\n4. TOTAL MISSING DATA (NaN + Unknown)")
total_nan = nan_counts.sum()
total_unknown = unknown_series.sum()
total_missing = total_nan + total_unknown
print(f"Total NaN: {total_nan}")
print(f"Total 'Unknown': {total_unknown}")
print(f"TOTAL MISSING: {total_missing}")
print(f"Percentage of dataset: {(total_missing / (df.shape[0] * df.shape[1])) * 100:.2f}%")

# ============================================================
# 5. COLUMNS WITH MOST MISSING DATA
# ============================================================
print("\n5. COLUMNS WITH MOST MISSING DATA")
combined_missing = {}
for col in df.columns:
    nan_count = df[col].isnull().sum()
    unknown_count = (df[col] == 'Unknown').sum() if df[col].dtype == 'object' else 0
    total = nan_count + unknown_count
    if total > 0:
        combined_missing[col] = total

combined_series = pd.Series(combined_missing).sort_values(ascending=False)
print(combined_series.head(10))

# ============================================================
# 6. CONSISTENCY CHECKS
# ============================================================
print("\n6. CONSISTENCY CHECKS")

# Check for negative values in numeric columns
print("\nNegative values in numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    negative = (df[col] < 0).sum()
    if negative > 0:
        print(f"  - {col}: {negative} negative values")

# Check for 'Unknown' in key categorical columns
print("\nCategorical columns with 'Unknown' values:")
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'str':
        unknown_count = (df[col] == 'Unknown').sum()
        if unknown_count > 0:
            print(f"  - {col}: {unknown_count} 'Unknown' values")

# ============================================================
# 7. OUTLIERS (IQR Method)
# ============================================================
print("\n7. OUTLIERS (IQR Method)")

# Define numeric columns to check for outliers
outlier_cols = ['energy_kcal', 'fat_100g', 'sugars_100g', 'proteins_100g', 
                'price_inr', 'serving_size_g']

for col in outlier_cols:
    if col in df.columns:
        # Convert to numeric, replace errors with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop NaN for calculation
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            print(f"  - {col}: {len(outliers)} outliers ({len(outliers)/len(col_data)*100:.2f}%)")
            print(f"    Range: {lower_bound:.2f} to {upper_bound:.2f}")

# ============================================================
# 8. SUMMARY OF DATA QUALITY
# ============================================================
print("\n8. SUMMARY OF DATA QUALITY")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Total NaN values: {total_nan}")
print(f"Total 'Unknown' values: {total_unknown}")
print(f"Total missing (NaN + Unknown): {total_missing}")
print(f"Percentage missing: {(total_missing / (df.shape[0] * df.shape[1])) * 100:.2f}%")

print("\n✅ Data quality check complete!")