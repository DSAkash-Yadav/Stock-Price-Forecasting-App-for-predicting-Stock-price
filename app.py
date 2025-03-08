import os
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Path to the model file
model_path = 'models/Stock Predictions Model.keras'

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path and try again.")
else:
    # Load the pre-trained model
    model = load_model(model_path)

    # Streamlit header
    st.set_page_config(page_title="Stock Market Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title('📈 Stock Market Predictor')
    

    # Sidebar for user inputs
    st.sidebar.header("Enter the Yahoo Finance symbol of the company whose data you want to analyze. By default, it is Google")
    stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
    start = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
    end = st.sidebar.date_input('End Date', pd.to_datetime('2022-12-31'))

    # Download stock data
    data = yf.download(stock, start, end)

    # Display stock data
    st.subheader('Stock Data')
    st.write(data)

    # Splitting the data into training and testing sets
    data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)

    if len(data_test) > 100:
        data_test_scale = scaler.fit_transform(data_test)
    else:
        st.error("Not enough test data to scale. Try increasing the date range.")

    # Layout with columns
    col1, col2 = st.columns(2)

    # Plotting the stock data
    with col1:
        st.subheader('📊 Stock Price and Volume')
        fig, ax = plt.subplots()
        data['Close'].plot(ax=ax, label='Close', grid=True)
        data['Volume'].plot(ax=ax, label='Volume', secondary_y=True)
        ax.legend()
        st.pyplot(fig)

    # Moving Averages
    with col2:
        st.subheader('📈 Price vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(data.Close, 'g', label='Close')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig1)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader('📉 Price vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(data.Close, 'g', label='Close')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig2)

    with col4:
        st.subheader('📊 Price vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r', label='MA100')
        plt.plot(ma_200_days, 'b', label='MA200')
        plt.plot(data.Close, 'g', label='Close')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig3)

    # Preparing the test data for prediction
    x, y = [], []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    # Predicting the stock prices
    if model:
        predict = model.predict(x)

        # Inversing the scaling
        scale = 1 / scaler.scale_
        predict = predict * scale
        y = y * scale

        # Plotting the original vs predicted prices using Plotly
        st.subheader('🔮 Original Price vs Predicted Price')

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            y=predict.flatten(), 
            x=data_test.index[100:], 
            mode='lines', 
            name='Predicted Price', 
            line=dict(color='firebrick', width=2)
        ))
        fig4.add_trace(go.Scatter(
            y=y, 
            x=data_test.index[100:], 
            mode='lines', 
            name='Original Price', 
            line=dict(color='forestgreen', width=2)
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
    else:
        st.error("Model could not be loaded. Please check the model path and try again.")
