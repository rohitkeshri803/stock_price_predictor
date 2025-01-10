# after feature 2:
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Download historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Perform Exploratory Data Analysis (EDA)
def plot_stock_price(stock_data):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Step 3: Feature Engineering
def add_features(stock_data):
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()  # Add 200-day moving average
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()  # Add 20-day exponential moving average
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()  # Add 20-day volatility
    stock_data = stock_data.dropna()  # Drop rows with NaN values
    return stock_data

# Step 4: Prepare data for training
def prepare_data(stock_data):
    features = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_50', 'MA_200', 'EMA_20', 'Volatility']]
    target = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 5: Perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_
    return best_model

# Step 6: Train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = hyperparameter_tuning(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = model.score(X_test, y_test)
    print(f'MAE: {mae}, RMSE: {rmse}, R^2: {r2}')
    return model, y_pred

# Step 7: Visualize predictions
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Main function
def main():
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2025-01-01'

    stock_data = get_stock_data(ticker, start_date, end_date)
    plot_stock_price(stock_data)
    stock_data = add_features(stock_data)
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_predictions(y_test, y_pred)

if __name__ == '__main__':
    main()
