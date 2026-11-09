import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from pathlib import Path

# ============================================================
# SETUP - Find the CSV file
# ============================================================

# Current directory
BASE_DIR = Path(__file__).resolve().parent

# Try to find the CSV in data/ folder
csv_path = BASE_DIR / "data" / "packaged_foods_india.csv"

if not csv_path.exists():
    # If not found, look in current directory
    csv_path = BASE_DIR / "packaged_foods_india.csv"

if not csv_path.exists():
    print("ERROR: Could not find packaged_foods_india.csv")
    print(f"Looked in: {csv_path}")
    exit()

print(f"Found dataset at: {csv_path}")

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv(csv_path)
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Clean numeric columns
numeric_cols = ['Calories_kcal', 'Carbohydrates_g', 'Proteins_g', 'Total_Fat_g', 
                'Saturated_Fat_g', 'Trans_Fat_g', 'Sugar_g', 'Sodium_mg', 
                'Dietary_Fiber_g', 'Cholesterol_mg', 'Calcium_mg', 'Iron_mg', 'Price_INR']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('<', '').str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Clean categorical columns
for col in ['Brand_Name', 'Category', 'Sub_Category']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

print("Data cleaned successfully!")

# ============================================================
# 2. EDA
# ============================================================

# Univariate - Distribution of Calories
plt.figure(figsize=(10, 5))
sns.histplot(df['Calories_kcal'], bins=30)
plt.title('Distribution of Calories (kcal)')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.savefig('eda_calories_distribution.png')
plt.show()
plt.close()

# Bivariate - Calories vs Total Fat
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Total_Fat_g', y='Calories_kcal', data=df)
plt.title('Total Fat vs Calories')
plt.xlabel('Total Fat (g)')
plt.ylabel('Calories (kcal)')
plt.savefig('eda_calories_vs_fat.png')
plt.show()
plt.close()

# Correlation Heatmap
num_cols = ['Calories_kcal', 'Carbohydrates_g', 'Proteins_g', 'Total_Fat_g', 'Sugar_g', 'Sodium_mg']
corr = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Nutrition Features')
plt.savefig('eda_correlation.png')
plt.show()
plt.close()

print("\nEDA plots saved as PNG files!")

# Remove extreme outliers (Calories > 1000)
df = df[df['Calories_kcal'] < 1000]
print(f"Dataset after removing outliers: {df.shape[0]} rows")
# ============================================================
# 3. MACHINE LEARNING MODELS
# ============================================================

# Features for ML
feature_cols = ['Carbohydrates_g', 'Proteins_g', 'Total_Fat_g', 'Sugar_g', 'Sodium_mg']
X = df[feature_cols]

# Classification target: High Fat (1 if fat > median, else 0)
y_class = (df['Total_Fat_g'] > df['Total_Fat_g'].median()).astype(int)

# Regression target: Calories
y_reg = df['Calories_kcal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification Models
print("\n" + "=" * 60)
print("CLASSIFICATION MODELS (Predict High Fat)")
print("=" * 60)

models = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name}: Accuracy = {acc:.4f}")

# Regression Models
print("\n" + "=" * 60)
print("REGRESSION MODELS (Predict Calories)")
print("=" * 60)

# Split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 6. Simple Linear Regression (use Carbohydrates only)
X_simple = X_train_reg_scaled[:, 0].reshape(-1, 1)
X_simple_test = X_test_reg_scaled[:, 0].reshape(-1, 1)
lin_simple = LinearRegression()
lin_simple.fit(X_simple, y_train_reg)
y_pred_simple = lin_simple.predict(X_simple_test)
print(f"Simple Linear Regression R²: {r2_score(y_test_reg, y_pred_simple):.4f}")

# 7. Multiple Linear Regression
lin_multi = LinearRegression()
lin_multi.fit(X_train_reg_scaled, y_train_reg)
y_pred_multi = lin_multi.predict(X_test_reg_scaled)
print(f"Multiple Linear Regression R²: {r2_score(y_test_reg, y_pred_multi):.4f}")

# 8. Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_reg_scaled)
X_test_poly = poly.transform(X_test_reg_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train_reg)
y_pred_poly = poly_reg.predict(X_test_poly)
print(f"Polynomial Regression R²: {r2_score(y_test_reg, y_pred_poly):.4f}")

print("\n✅ All done! Check the PNG files and results above.")