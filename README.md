# Stock Price Predictor

## This project predicts stock prices using machine learning techniques such as Linear Regression, Ridge, and Lasso regression. It uses historical stock data to train a model and make predictions about future stock prices.
### Project Overview

    Objective: Predict stock prices using historical stock data.
    Techniques Used:
        Linear Regression
        Ridge Regression
        Lasso Regression
        Hyperparameter tuning using GridSearchCV
        Exploratory Data Analysis (EDA)
        Feature engineering with moving averages, exponential moving averages, and volatility

### Prerequisites

Before running this project, ensure you have the following libraries installed:

    yfinance
    pandas
    numpy
    matplotlib
    sklearn

You can install the required libraries using pip:

pip install yfinance pandas numpy matplotlib scikit-learn

Steps
1. Download Historical Stock Data

The get_stock_data() function fetches historical stock data for a given ticker symbol (e.g., 'AAPL') between specified start and end dates using the yfinance library.
2. Perform Exploratory Data Analysis (EDA)

The plot_stock_price() function visualizes the stock's closing price over time.
3. Feature Engineering

The add_features() function generates the following additional features:

    50-day moving average
    200-day moving average
    20-day exponential moving average
    Daily returns (percentage change)
    20-day volatility (rolling standard deviation of daily returns)

4. Prepare Data for Training

The prepare_data() function splits the dataset into training and testing sets, with features like Open, High, Low, Volume, and the engineered features, and the target being the Close price.
5. Hyperparameter Tuning

The hyperparameter_tuning() function uses GridSearchCV to perform cross-validation and find the best hyperparameters for Ridge Regression. The parameter alpha is tuned to improve model performance.
6. Train and Evaluate the Model

The train_and_evaluate() function trains the model using the best parameters found during hyperparameter tuning and evaluates it on the test set using metrics like:

    Mean Absolute Error (MAE)
    Root Mean Squared Error (RMSE)
    R-squared (R²)

7. Visualize Predictions

The plot_predictions() function compares the actual stock prices with the predicted prices visually.
Example Usage

if __name__ == '__main__':
    main()

This will run the program for the stock ticker 'AAPL' between the start date '2015-01-01' and end date '2025-01-01'. It will:

    Download the stock data
    Plot the stock price
    Add technical features
    Train and evaluate the model
    Plot actual vs predicted stock prices

Output

The program will output:

    The best hyperparameters for Ridge regression
    The model evaluation metrics: MAE, RMSE, and R²
    A graph comparing the actual stock prices with the predicted prices

License

This project is open source and available under the MIT License.
