# House Price Prediction – Internship Assignment

This project predicts house sale prices using advanced regression techniques. It is part of an internship assignment to demonstrate skills in data preprocessing, feature engineering, model training, and prediction generation.

---


---

##  Objective

> Predict the final sale price of residential homes using data from the Ames Housing dataset ([Kaggle Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)).

---

## ⚙️ Workflow Overview

### 1. Data Preprocessing
- Removed high-missing or irrelevant features (e.g., `PoolQC`, `Fence`)
- Imputed numerical columns using median, categorical with mode or `"None"`

### 2. Feature Engineering
Created domain-informed features to capture better patterns:
- `OverallQual_GrLivArea`: Combines size and quality
- `TotalSF`: Total square footage (above + below ground)
- `TotalBath`: Full + half + basement baths
- `Qual_Age_Interaction`: Newer, high-quality homes = higher price
- `HouseAge`, `YearsSinceRemodel`, and log-transformed skewed features

### 3. Model Training
Compared two models:
| Model             | RMSE  | R² Score |
|------------------|------------------|----------------------|
| Linear Regression | 35,484           | 0.836                |
| **Random Forest** | **29,749**     | **0.885**          |

---

##  Why Random Forest Was Chosen

-  **Captures non-linear relationships** between features and price
-  **Automatic feature interaction handling** (no need to manually combine all features)
-  **Robust to outliers** and missing values
-  **No need for feature scaling**
-  **Significantly improved accuracy over Linear Regression**

---

##  Files Explained

- `house-price-prediction.ipynb` – Full notebook: EDA → preprocessing → feature engineering → model training → evaluation
- `house_price_model.pkl` – Trained model saved using `joblib`
- `generate_prediction.py` – Python script to:
  - Load `test.csv`
  - Apply same feature engineering
  - Load the trained model
  - Predict `SalePrice`
  - Save `prediction.csv`
- `prediction.csv` – Output file with predictions (columns: `Id`, `SalePrice`)

---

## 🚀 How to Run the Project

1. Make sure `train.csv`, `test.csv`, and the notebook are inside the `data/` and `notebook/` folders respectively.
2. Run all cells in `notebook/house-price-prediction.ipynb` to regenerate model if needed.
3. Then run:

```bash
python generate_prediction.py


