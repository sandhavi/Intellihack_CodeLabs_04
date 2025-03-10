import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle

# Load the trained LSTM model
model = load_model('models/improved_lstm_model.keras')

# Load the scalers
with open('scalers/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('scalers/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Function to preprocess input data
def preprocess_data(df):
    features = ['Open', 'High', 'Low', 'Close_Lag_1', 'MA_10', 'BB_Upper', 'MA_5', 'Cumulative_Return', 'Adj Close', 'Close']
    X = df[features].values
    X_scaled = scaler_X.transform(X)

    sequence_length = 60
    X_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])

    return np.array(X_sequences)

# Predict stock prices
def predict_stock_price(data):
    X_sequences = preprocess_data(data)
    y_pred_scaled = model.predict(X_sequences)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred

# Function to analyze trends
def detect_trend(df):
    if df['Close'].iloc[-1] > df['Close'].iloc[-10]:
        return "📈 Uptrend"
    elif df['Close'].iloc[-1] < df['Close'].iloc[-10]:
        return "📉 Downtrend"
    else:
        return "🔄 Sideways"

# Function to calculate Bollinger Bands & Volatility
def calculate_volatility(df):
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['20_STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['20_MA'] + (df['20_STD'] * 2)
    df['BB_Lower'] = df['20_MA'] - (df['20_STD'] * 2)
    return df

# Function to detect support & resistance levels
def support_resistance_levels(df):
    support = df['Low'].rolling(window=20).min().iloc[-1]
    resistance = df['High'].rolling(window=20).max().iloc[-1]
    return support, resistance

# Function to calculate MACD & RSI
def trading_signals(df):
    df['12_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Price_Change'] = df['Close'].diff(1)
    df['Gain'] = np.where(df['Price_Change'] > 0, df['Price_Change'], 0)
    df['Loss'] = np.where(df['Price_Change'] < 0, -df['Price_Change'], 0)
    
    df['Avg_Gain'] = df['Gain'].rolling(window=14).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=14).mean()
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
    macd_signal = "🟢 BUY Signal" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "🔴 SELL Signal"
    rsi_signal = "🟢 BUY (Oversold)" if df['RSI'].iloc[-1] < 30 else "🔴 SELL (Overbought)" if df['RSI'].iloc[-1] > 70 else "⚪ Neutral"
    
    return macd_signal, rsi_signal

# Function for ROI prediction
def calculate_roi(df, predicted_prices, investment):
    initial_price = df['Close'].iloc[-1]
    future_price = predicted_prices[-1][0]
    return ((future_price - initial_price) / initial_price) * 100, (investment * (future_price / initial_price))

# Streamlit UI
st.title('📊 Stock Market Price Prediction & Analysis')

st.sidebar.header('Upload Stock Data')
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = calculate_volatility(df)

    st.subheader("📉 Uploaded Stock Data")
    st.write(df.tail())

    # Ensure 'Close' column exists for visualization
    if 'Close' in df.columns:
        last_close_price = df['Close'].iloc[-1]
        st.metric(label="Last Known Close Price", value=f"${last_close_price:.2f}")

    # Predict future prices
    predicted_price = predict_stock_price(df)

    # Analyze trend
    trend = detect_trend(df)

    # Calculate support & resistance
    support, resistance = support_resistance_levels(df)

    # Generate trading signals
    macd_signal, rsi_signal = trading_signals(df)

    # Investment ROI Calculation
    investment_amount = st.sidebar.number_input("💰 Investment Amount ($)", min_value=100, value=1000)
    roi_percentage, projected_value = calculate_roi(df, predicted_price, investment_amount)

    # Display predictions
    st.subheader("🔮 Predicted Stock Prices")
    st.write(predicted_price)

    # Display Trend Analysis
    st.subheader("📊 Market Trend")
    st.write(trend)

    # Display Volatility Analysis
    st.subheader("📈 Bollinger Bands & Volatility")
    st.line_chart(df[['Close', 'BB_Upper', 'BB_Lower']])

    # Display Support & Resistance Levels
    st.subheader("📌 Support & Resistance Levels")
    st.write(f"📉 **Support Level:** ${support:.2f}")
    st.write(f"📈 **Resistance Level:** ${resistance:.2f}")

    # Display Trading Signals
    st.subheader("📍 Trading Signals")
    st.write(f"**MACD Signal:** {macd_signal}")
    st.write(f"**RSI Signal:** {rsi_signal}")

    # Display ROI Prediction
    st.subheader("💰 Investment ROI Projection")
    st.write(f"💵 If you invest **${investment_amount}** today, your estimated future value is **${projected_value:.2f}**")
    st.write(f"📈 Expected Return: **{roi_percentage:.2f}%**")

    # Show Interactive Price Prediction Graph
    st.subheader("📉 Future Price Prediction Visualization")
    fig, ax = plt.subplots()
    ax.plot(predicted_price, label="Predicted Close Price", linestyle="dashed", color="blue")
    ax.set_title("Stock Price Forecast")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a stock dataset to proceed!")
