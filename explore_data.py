import pandas as pd

# Load the training data
df = pd.read_csv('data/train.csv')

# Show basic info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column (top 20):")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(20))
# Check if these features exist
features_needed = ['Neighborhood', 'GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr',
                   'FullBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces',
                   'GarageCars', 'GarageArea', 'CentralAir', 'SalePrice']
for col in features_needed:
    if col in df.columns:
        print(f"✅ {col} found")
    else:
        print(f"❌ {col} NOT found")