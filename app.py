import yfinance as yf

# Fetch stock data
stock_data = yf.download('AAPL', start='2015-01-01', end='2023-12-31')
print(stock_data.head())

import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

# Extract the 'Close' prices
data = stock_data[['Close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

import numpy as np

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])  # Past 60 days
        y.append(data[i, 0])              # Next day's price
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50)

# Reshape data for prediction
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Rescale back to original range

# Compare with actual prices
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[train_size + 60:], y_test, color='blue', label='Actual Prices')
plt.plot(stock_data.index[train_size + 60:], predictions, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


