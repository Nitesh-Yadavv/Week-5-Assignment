import joblib
import pandas as pd

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    # Engineered features
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_Age_Interaction'] = df['OverallQual'] * (2025 - df['YearBuilt'])
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['GrLivArea']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['TotalBath']
    df['HouseAge'] = 2025 - df['YearBuilt']
    df['YearsSinceRemodel'] = 2025 - df['YearRemodAdd']

    # Select final features used by the model
    selected_features = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
        '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
        'OverallQual_GrLivArea', 'Qual_Age_Interaction', 'TotalSF',
        'TotalBath', 'TotalRooms', 'HouseAge', 'YearsSinceRemodel'
    ]

    return df[selected_features]


def predict_house_price(input_data):
    # Load model
    model = joblib.load("notebook/house_price_model.pkl")

    # Preprocess input
    processed_data = preprocess_input(input_data)

    # Predict
    prediction = model.predict(processed_data)
    return prediction[0]


if __name__ == "__main__":
    # Example input (you can change these values)
    sample_input = {
        'OverallQual': 7,
        'GrLivArea': 1800,
        'GarageCars': 2,
        'TotalBsmtSF': 1000,
        '1stFlrSF': 1100,
        'FullBath': 2,
        'HalfBath': 1,
        'BsmtFullBath': 1,
        'BsmtHalfBath': 0,
        'TotRmsAbvGrd': 6,
        'YearBuilt': 2010,
        'YearRemodAdd': 2015
    }

    predicted_price = predict_house_price(sample_input)
    print("\nPredicted House Sale Price: $", round(predicted_price, 2))
