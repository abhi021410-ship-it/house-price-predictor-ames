# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# ----- 1. Load the data -----
print("Loading data from data/train.csv ...")
df = pd.read_csv('data/train.csv')

# ----- 2. Select features (location, size, rooms, amenities) -----
features = [
    'Neighborhood',      # location
    'GrLivArea',         # above ground living area (sqft)
    'TotalBsmtSF',       # total basement area (sqft)
    'BedroomAbvGr',      # bedrooms above ground
    'FullBath',          # full bathrooms
    'HalfBath',          # half bathrooms
    'KitchenAbvGr',      # kitchens above ground
    'Fireplaces',        # number of fireplaces
    'GarageCars',        # garage capacity (cars)
    'GarageArea',        # garage area (sqft)
    'CentralAir'         # central air conditioning (Y/N)
]

target = 'SalePrice'

# Keep only the relevant columns and drop rows where SalePrice is missing
df = df[features + [target]].dropna(subset=[target])

X = df[features]
y = df[target]

print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features")

# ----- 3. Define preprocessing -----
numeric_features = [
    'GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath',
    'HalfBath', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'GarageArea'
]
categorical_features = ['Neighborhood', 'CentralAir']

# For numbers: fill missing with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categories: fill missing with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ----- 4. Build the full pipeline -----
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ----- 5. Split, train, and evaluate -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training the model...")
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.3f}")
print(f"Test R²:  {test_score:.3f}")

# ----- 6. Save everything -----
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(preprocessor, 'model/preprocessor.pkl')
joblib.dump(features, 'model/feature_names.pkl')
print("Model saved in 'model/' folder:")
print("  - model.pkl")
print("  - preprocessor.pkl")
print("  - feature_names.pkl")