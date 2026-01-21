import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold/Silver AI (TensorFlow)", layout="wide")

st.title("🇮🇳 Gold & Silver Analytics (TensorFlow LSTM)")
st.markdown("""
This dashboard uses a **Deep Learning LSTM Model** to predict prices.
* **Gold:** Price per **10 Grams**
* **Silver:** Price per **1 Kilogram**
""")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
metal_choice = st.sidebar.selectbox("Select Asset:", ["Gold", "Silver"])
period = st.sidebar.selectbox("Data Period:", ["1y", "2y", "5y", "max"], index=1)

metal_ticker = "GC=F" if metal_choice == "Gold" else "SI=F"
currency_ticker = "USDINR=X"

# --- DATA PREP ---
def fix_data_structure(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data
def load_and_convert_data(metal_sym, curr_sym, period):
    metal_data = yf.download(metal_sym, period=period, progress=False)
    metal_data = fix_data_structure(metal_data)
    
    curr_data = yf.download(curr_sym, period=period, progress=False)
    curr_data = fix_data_structure(curr_data)
    
    if metal_data.empty: return pd.DataFrame()

    df = metal_data.copy()
    aligned_currency = curr_data['Close'].reindex(df.index).ffill().bfill()
    
    if metal_choice == "Gold":
        factor = 10 / 31.1035
    else:
        factor = 1 / 0.0311035

    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] * aligned_currency) * factor
        
    df = df.dropna()
    df.reset_index(inplace=True)
    return df

try:
    data = load_and_convert_data(metal_ticker, currency_ticker, period)
    if data.empty:
        st.error("Data unavailable. Try a different period.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- TENSORFLOW HELPER FUNCTIONS ---
def prepare_lstm_data(data, lookback=60):
    # We only use 'Close' price for simplicity in this LSTM
    dataset = data['Close'].values.reshape(-1, 1)
    
    # Scale data between 0 and 1 (Crucial for LSTM)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    
    # Create sequences: Use past 60 days to predict next day
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

# --- TABS ---
tab1, tab2 = st.tabs(["📈 Market Dashboard", "🧠 TensorFlow Prediction"])

with tab1:
    unit = "10g" if metal_choice == "Gold" else "1kg"
    current_price = data['Close'].iloc[-1]
    
    st.metric(f"Current Price (₹/{unit})", f"₹{current_price:,.0f}")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='Price'))
    fig.update_layout(title=f'{metal_choice} Price Trend', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader(f"🧠 LSTM Deep Learning Model")
    st.write("This model uses a **Recurrent Neural Network (LSTM)**. It looks at the last **60 days** of prices to predict the next movement.")

    if len(data) > 100:
        with st.spinner("Training Neural Network... (This may take a moment)"):
            try:
                # Prepare Data
                look_back = 60
                x_train, y_train, scaler, raw_scaled = prepare_lstm_data(data, lookback=look_back)
                
                # Build LSTM Model
                model = Sequential()
                # Layer 1: LSTM with 50 neurons
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                # Layer 2: LSTM with 50 neurons
                model.add(LSTM(units=50, return_sequences=False))
                # Layer 3: Dense (Output)
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                
                # Compile & Train
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
                
                # Predict Next Day
                # Get last 60 days from raw data
                last_60_days = raw_scaled[-look_back:]
                last_60_days = last_60_days.reshape(1, look_back, 1)
                
                pred_scaled = model.predict(last_60_days)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                
                # Display Results
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.write("### Predicted Price Tomorrow")
                    st.success(f"# ₹{pred_price:,.0f}")
                
                with c2:
                    diff = pred_price - current_price
                    st.write("### Expected Movement")
                    if diff > 0:
                        st.write(f"📈 **UP** by ₹{abs(diff):,.0f}")
                    else:
                        st.write(f"📉 **DOWN** by ₹{abs(diff):,.0f}")
                        
                st.caption(f"Model trained on {len(x_train)} data sequences over 5 epochs.")
                
            except Exception as e:
                st.error(f"Modeling Error: {e}")
    else:
        st.warning("Needs at least 100 days of data to train LSTM. Select '1y' or 'max'.")
