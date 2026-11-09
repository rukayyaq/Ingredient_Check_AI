import pandas as pd
import numpy as np
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

# Load fused dataset
base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "data" / "new_fused_dataset.csv")

print(f"Dataset shape: {df.shape}")

# ============================================================
# PREPROCESSING: CLEAN COLUMNS
# ============================================================

# Clean numeric columns
numeric_cols = ['energy_kcal', 'fat_100g', 'sugars_100g', 'proteins_100g', 
                'carbohydrates_100g', 'sodium_mg']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].replace('Unknown', np.nan)
        df[col] = df[col].astype(str).str.replace('<', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# ============================================================
# CLASSIFICATION: Predict High Fat
# ============================================================

# Features for classification (CAN include energy_kcal)
feature_cols_class = ['energy_kcal', 'fat_100g', 'sugars_100g', 'proteins_100g', 
                      'carbohydrates_100g', 'sodium_mg']
X_class = df[feature_cols_class]

# Create binary target (1 if fat > median, else 0)
y_class = (df['fat_100g'] > df['fat_100g'].median()).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Scale features
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

# ============================================================
# REGRESSION: Predict Calories (DO NOT include energy_kcal as feature!)
# ============================================================

# Features for regression (EXCLUDE energy_kcal)
feature_cols_reg = ['fat_100g', 'sugars_100g', 'proteins_100g', 
                    'carbohydrates_100g', 'sodium_mg']
X_reg = df[feature_cols_reg]

# Target: Calories
y_reg = df['energy_kcal']

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("\n" + "=" * 60)
print("REGRESSION MODELS (Predict Calories)")
print("=" * 60)

# 6. Simple Linear Regression (use first feature - Fat)
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

print("\n✅ ML models complete!")