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
st.set_page_config(page_title="Gold/Silver AI Insight", layout="wide", page_icon="✨")

st.title("✨ Gold & Silver Market Intelligence")
st.markdown("### Real-time prices, AI forecasts, and market news for India 🇮🇳")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
metal_choice = st.sidebar.selectbox("Select Asset:", ["Gold", "Silver"])

# Indian States List
indian_states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", 
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", 
    "Ladakh", "Lakshadweep", "Puducherry"
]
selected_state = st.sidebar.selectbox("Your Location:", indian_states)

# Tax Toggle
tax_option = st.sidebar.radio("Price Display:", ["Exclude Tax", "Include GST (3%)"])

period = st.sidebar.selectbox("History:", ["1y", "2y", "5y", "max"], index=1)

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
    
    # Apply Tax if Selected
    if tax_option == "Include GST (3%)":
        data[['Open', 'High', 'Low', 'Close']] *= 1.03

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- AI HELPER FUNCTIONS ---
def prepare_lstm_data(data, lookback=60):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Market Dashboard", "🔮 AI Forecast", "📰 Latest News"])

with tab1:
    unit = "10g" if metal_choice == "Gold" else "1kg"
    current_price = data['Close'].iloc[-1]
    last_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = current_price - last_close
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Current Price")
        if tax_option == "Include GST (3%)":
            st.caption(f"Price in {selected_state} (Inc. GST)")
        else:
            st.caption(f"Price in {selected_state} (Excl. Tax)")
            
        st.metric(label=f"₹/{unit}", value=f"₹{current_price:,.0f}", delta=f"{change:,.0f} ₹")
        
    with col2:
        st.subheader("Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name='Price'))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"🔮 AI Price Prediction")
    st.write(f"Our Artificial Intelligence analyzes the past **60 days** of market data to forecast where {metal_choice} prices might go tomorrow.")

    if len(data) > 100:
        if st.button("Generate Forecast"):
            with st.spinner("Analyzing market patterns..."):
                try:
                    # Prepare Data
                    look_back = 60
                    x_train, y_train, scaler, raw_scaled = prepare_lstm_data(data, lookback=look_back)
                    
                    # Build Model (Simplified for user, same strength)
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dense(units=25))
                    model.add(Dense(units=1))
                    
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
                    
                    # Predict Next Day
                    last_60_days = raw_scaled[-look_back:]
                    last_60_days = last_60_days.reshape(1, look_back, 1)
                    
                    pred_scaled = model.predict(last_60_days)
                    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                    
                    # Display Results
                    st.divider()
                    st.succes("AI Analysis Complete!")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("### Tomorrow's Forecast")
                        st.metric(label="Predicted Price", value=f"₹{pred_price:,.0f}")
                    
                    with c2:
                        diff = pred_price - current_price
                        st.write("### Expected Trend")
                        if diff > 0:
                            st.success(f"📈 UP by ₹{abs(diff):,.0f}")
                        else:
                            st.error(f"📉 DOWN by ₹{abs(diff):,.0f}")
                            
                    st.info("Note: AI predictions are based on historical patterns and should not be the sole basis for financial decisions.")
                    
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
    else:
        st.warning("Needs at least 100 days of data for accurate AI analysis. Please select '1y' or 'max' in the sidebar.")

with tab3:
    st.subheader(f"📰 Latest {metal_choice} News")
    
    try:
        ticker = yf.Ticker(metal_ticker)
        news_list = ticker.news
        
        if news_list:
            for item in news_list:
                with st.container():
                     # Extract info safely
                    content = item.get('content', {})
                    title = content.get('title', 'No Title')
                    # Try to find a link
                    if 'clickThroughUrl' in content and content['clickThroughUrl']:
                         link = content['clickThroughUrl']['url']
                    elif 'canonicalUrl' in content and content['canonicalUrl']:
                         link = content['canonicalUrl']['url']
                    else:
                         link = "#"
                         
                    summary = content.get('summary', 'No summary available.')
                    provider = content.get('provider', {}).get('displayName', 'Unknown Source')
                    pub_date = content.get('pubDate', '')
                    
                    st.markdown(f"### [{title}]({link})")
                    st.caption(f"Source: {provider} | {pub_date}")
                    st.write(summary)
                    st.divider()
        else:
            st.write("No direct news feed available at the moment.")
    except Exception as e:
        st.error("Could not fetch live news. Please try again later.")
        st.caption(str(e))
