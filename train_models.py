import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
import os

# Create directory to save models
os.makedirs("models", exist_ok=True)

# Load and preprocess data
def preprocess_data(df):
    df.dropna(inplace=True)
    df['Elap Time'] = pd.to_timedelta(df['Elap Time'].astype(str), errors='coerce')
    df['Seconds'] = df['Elap Time'].dt.total_seconds()
    df.dropna(inplace=True)
    df.set_index('Seconds', inplace=True)
    return df

# Train and save models for a region
def train_and_save_models(df, region):
    df = preprocess_data(df)
    X = df.index.values.reshape(-1, 1)
    y = df['Swell (%)'].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"models/{region}_rf.pkl")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, f"models/{region}_lr.pkl")

    # ARIMA (only on y series)
    model = ARIMA(y, order=(2,1,2))
    arima_result = model.fit()
    joblib.dump(arima_result, f"models/{region}_arima.pkl")

    # Symbolic Regression
    symb = SymbolicRegressor(population_size=500, generations=20,
                             stopping_criteria=0.01, p_crossover=0.7,
                             p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                             p_point_mutation=0.1, max_samples=0.9,
                             verbose=1, parsimony_coefficient=0.01,
                             random_state=42)
    symb.fit(X_train, y_train)
    joblib.dump(symb, f"models/{region}_symbolic.pkl")

# Load Excel file
file_path = r'D:\University\Research Paper\Sixth\different shales.xlsx'
data = pd.read_excel(file_path, sheet_name=None)

for region, df in data.items():
    train_and_save_models(df, region)
print("Models trained and saved.")
