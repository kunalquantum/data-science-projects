import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_option('deprecation.showfileUploaderEncoding', False)

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def add_features(stock_data, lag_days=[1, 3, 7], moving_averages=[7, 30]):
    for lag in lag_days:
        stock_data[f'Close_Lag_{lag}'] = stock_data['Close'].shift(lag)

    for window in moving_averages:
        stock_data[f'MA_{window}'] = stock_data['Close'].rolling(window=window).mean()
       # Drop rows with missing values in the added features
    stock_data.dropna(subset=[f'Close_Lag_{lag}' for lag in lag_days] + [f'MA_{window}' for window in moving_averages], inplace=True)

    return stock_data

def add_next_day_close(stock_data):
    stock_data['Next Day Close'] = stock_data['Close'].shift(-1)
    stock_data.dropna(subset=['Next Day Close'], inplace=True)
    return stock_data

def predict_next_day_close(stock_data):
    X = stock_data[['Close', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7', 'MA_7', 'MA_30']].values
    y = stock_data['Next Day Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    st.write(f"Mean Squared Error: {mse}")

    return model, predictions

st.title(" Stock Price Analysis and Prediction App")

ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD):", "2022-01-01")
end_date = st.sidebar.text_input("Enter End Date (YYYY-MM-DD):", "2023-01-01")

stock_data = get_stock_data(ticker, start_date, end_date)
stock_data = add_features(stock_data)
stock_data = add_next_day_close(stock_data)
st.write(f"## {ticker} Stock Data")
st.write(stock_data.head())

model, predictions = predict_next_day_close(stock_data)

st.write("## Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_data.index[-len(predictions):], stock_data['Close'][-len(predictions):], label='Actual Closing Prices', color='blue')
ax.plot(stock_data.index[-len(predictions):], predictions, label='Predicted Closing Prices', color='orange')
ax.legend()
st.pyplot(fig)
