import pandas as pd
import joblib

file_path = "data/test.csv" 
raw_test = pd.read_csv(file_path)

def feature_engineering(df):
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_Age_Interaction'] = df['OverallQual'] * (2025 - df['YearBuilt'])
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['GrLivArea']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['TotalBath']
    df['HouseAge'] = 2025 - df['YearBuilt']
    df['YearsSinceRemodel'] = 2025 - df['YearRemodAdd']
    return df

processed_test = feature_engineering(raw_test.copy())

selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'OverallQual_GrLivArea', 'Qual_Age_Interaction', 'TotalSF',
    'TotalBath', 'TotalRooms', 'HouseAge', 'YearsSinceRemodel'
]

model = joblib.load("notebook/house_price_model.pkl")

X_final_test = processed_test[selected_features]
predictions = model.predict(X_final_test)

submission = pd.DataFrame({
    "Id": raw_test["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created!")
