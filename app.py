import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Alpha Vantage API key
API_KEY = " RDDU0R45ABGUU5ZX"

# Load the pre-trained model
model_path = 'models/Stock Predictions Model.keras'
model = load_model(model_path)

# Streamlit setup
st.set_page_config(layout="wide", page_title="📈 Stock Market Predictor")
st.title('📈 Stock Market Predictor')

# Sidebar for user inputs
st.sidebar.header('User Inputs')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol', 'GOOG', help='For example: GOOG, AAPL, MSFT')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2022-12-31'))

# Validate date range
if start_date >= end_date:
    st.error("Start date must be earlier than the end date. Please adjust the date range.")
    st.stop()

# Fetch stock data using Alpha Vantage
st.subheader('Fetching Stock Data...')
try:
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    stock_data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
    stock_data = stock_data.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume",
    })
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data = stock_data.sort_index()

    # Filter data by date range
    stock_data = stock_data.loc[start_date:end_date]
    if stock_data.empty:
        st.error(f"No data found for the stock symbol '{stock_symbol}' in the specified date range. Please adjust the range.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred while fetching stock data: {e}")
    st.stop()

# Display stock data
st.subheader('Stock Data')
st.write(stock_data)

# Split data into training and testing sets
data_train = stock_data['Close'][:int(len(stock_data) * 0.80)]
data_test = stock_data['Close'][int(len(stock_data) * 0.80):]

# Data scaling
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)

if len(data_test_combined) > 100:
    scaled_data_test = scaler.fit_transform(data_test_combined.values.reshape(-1, 1))
else:
    st.error("Not enough data to scale for predictions. Try increasing the date range.")
    st.stop()

# Layout with columns
col1, col2 = st.columns(2)

# Plotting the stock data
with col1:
    st.subheader('📊 Stock Price and Volume')
    fig, ax = plt.subplots()
    stock_data['Close'].plot(ax=ax, label='Close', grid=True, color='blue')
    stock_data['Volume'].plot(ax=ax, label='Volume', secondary_y=True, color='orange')
    ax.legend(loc='best')
    st.pyplot(fig)

# Moving Averages
with col2:
    st.subheader('📈 Price vs MA50')
    ma_50_days = stock_data['Close'].rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(stock_data['Close'], 'g', label='Close')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig1)

col3, col4 = st.columns(2)
with col3:
    st.subheader('📉 Price vs MA50 vs MA100')
    ma_100_days = stock_data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(ma_100_days, 'b', label='MA100')
    plt.plot(stock_data['Close'], 'g', label='Close')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig2)

with col4:
    st.subheader('📊 Price vs MA100 vs MA200')
    ma_200_days = stock_data['Close'].rolling(200).mean()
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='MA100')
    plt.plot(ma_200_days, 'b', label='MA200')
    plt.plot(stock_data['Close'], 'g', label='Close')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig3)

# Preparing the test data for prediction
x_test, y_test = [], []
for i in range(100, len(scaled_data_test)):
    x_test.append(scaled_data_test[i-100:i])
    y_test.append(scaled_data_test[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting the stock prices
predictions = model.predict(x_test)

# Inverse scaling
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Plotting the original vs predicted prices
st.subheader('🔮 Original Price vs Predicted Price')

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    y=predictions.flatten(),
    x=data_test_combined.index[100:],
    mode='lines',
    name='Predicted Prices',
    line=dict(color='blue', width=2)
))
fig4.add_trace(go.Scatter(
    y=y_test,
    x=data_test_combined.index[100:],
    mode='lines',
    name='Original Prices',
    line=dict(color='green', width=2)
))

fig4.update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    height=600,
    width=1200,
    xaxis=dict(showgrid=True, gridcolor='LightGray'),
    yaxis=dict(showgrid=True, gridcolor='LightGray'),
    plot_bgcolor='white',
    hovermode='x unified'
)

st.plotly_chart(fig4)