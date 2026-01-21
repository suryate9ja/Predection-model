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
st.set_page_config(page_title="Gold & Silver Analytics", layout="wide", page_icon="None")

# Custom CSS for "Premium Glass" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global Settings */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1a1a 0%, #000000 100%);
        color: #ffffff;
    }

    /* Glass Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .metric-value {
        font-size: 2.5em;
        font-weight: 300;
        letter-spacing: -1px;
        color: #ffffff;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #888888;
        font-weight: 600;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: transparent;
        color: #888888;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: transparent !important;
        border-bottom: 2px solid #ffffff;
        font-weight: 600;
    }

    /* Clean Streamlit Components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

st.title("Market Overview")
st.markdown("Live Prices & Analytics for Indian Markets")

# --- SIDEBAR ---
st.sidebar.markdown("### PREFERENCES")

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
selected_state = st.sidebar.selectbox("Location", indian_states)

# Tax Toggle
tax_option = st.sidebar.radio("Display Mode", ["Base Price", "Include GST (3%)"])

period = st.sidebar.selectbox("TIMEFRAME", ["1y", "2y", "5y", "max"], index=1)

# API Tickers
gold_ticker = "GC=F"
silver_ticker = "SI=F"
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
def load_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    return fix_data_structure(data)

@st.cache_data
def get_market_data(period):
    # Load all raw data
    raw_gold = load_data(gold_ticker, period)
    raw_silver = load_data(silver_ticker, period)
    raw_forex = load_data(currency_ticker, period)
    
    if raw_gold.empty or raw_silver.empty or raw_forex.empty:
        return None, None

    # Align dates
    common_index = raw_gold.index.intersection(raw_silver.index).intersection(raw_forex.index)
    
    gold = raw_gold.loc[common_index].copy()
    silver = raw_silver.loc[common_index].copy()
    forex = raw_forex['Close'].loc[common_index]
    
    # Conversion Factors
    # Gold: Troy Oz -> 10 Grams (1 Troy Oz = 31.1035 Grams)
    gold_factor = 10 / 31.1035
    # Silver: Troy Oz -> 1 Kg (1 Troy Oz = 0.0311035 Kg)
    silver_factor = 1 / 0.0311035
    
    # Calculate INR Prices
    gold['INR_Close'] = gold['Close'] * forex * gold_factor
    silver['INR_Close'] = silver['Close'] * forex * silver_factor
    
    return gold, silver

try:
    gold_df, silver_df = get_market_data(period)
    
    if gold_df is None:
        st.error("Data unavailable. Please verify your connection.")
        st.stop()
        
    # Apply Tax Logic
    tax_multiplier = 1.03 if tax_option == "Include GST (3%)" else 1.0
    
    current_gold_24k = gold_df['INR_Close'].iloc[-1] * tax_multiplier
    current_silver_1kg = silver_df['INR_Close'].iloc[-1] * tax_multiplier
    
    # 22K Gold Calculation (Standard: 91.6% of 24K)
    current_gold_22k = current_gold_24k * 0.916

    # Previous Close for Delta
    prev_gold_24k = gold_df['INR_Close'].iloc[-2] * tax_multiplier
    prev_silver_1kg = silver_df['INR_Close'].iloc[-2] * tax_multiplier
    
    delta_gold = current_gold_24k - prev_gold_24k
    delta_silver = current_silver_1kg - prev_silver_1kg

except Exception as e:
    st.error(f"Error loading market data: {e}")
    st.stop()

# --- DASHBOARD UI ---

# 1. Key Metrics Row
col1, col2, col3 = st.columns(3)

def display_card(col, title, price, delta, unit):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">₹{price:,.0f}</div>
            <div style="color: {'#4CAF50' if delta > 0 else '#FF5252'}; font-size: 0.9em;">
                {("+" if delta > 0 else "")}{delta:,.0f}
            </div>
            <div style="font-size: 0.7em; color: #666; margin-top: 5px;">PER {unit.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

display_card(col1, "Gold 24K", current_gold_24k, delta_gold, "10g")
display_card(col2, "Gold 22K", current_gold_22k, delta_gold * 0.916, "10g")
display_card(col3, "Silver", current_silver_1kg, delta_silver, "1kg")

st.markdown("---")

# 2. Tabs for Charts & Advanced Features
main_tab1, main_tab2, main_tab3 = st.tabs(["Prices", "AI Forecast", "News"])

with main_tab1:
    chart_view = st.radio("Select View:", ["Gold Trend", "Silver Trend"], horizontal=True)
    
    fig = go.Figure()
    
    if chart_view == "Gold Trend":
        # Plotting 24K Gold
        # Recalculate series with tax
        plot_data = gold_df['INR_Close'] * tax_multiplier
        fig.add_trace(go.Scatter(x=gold_df.index, y=plot_data, mode='lines', name='Gold 24K', line=dict(color='#FFD700', width=2)))
        fig.update_layout(title="Gold Price History (24K / 10g)", yaxis_title="Price (INR)", template="plotly_dark")
    else:
        # Plotting Silver
        plot_data = silver_df['INR_Close'] * tax_multiplier
        fig.add_trace(go.Scatter(x=silver_df.index, y=plot_data, mode='lines', name='Silver', line=dict(color='#C0C0C0', width=2)))
        fig.update_layout(title="Silver Price History (1Kg)", yaxis_title="Price (INR)", template="plotly_dark")
        
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- AI HELPER FUNCTIONS ---
def prepare_lstm_data(series, lookback=60):
    dataset = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

with main_tab2:
    st.subheader("Artificial Intelligence Forecast")
    st.write("Our AI analyzes global market patterns to predict tomorrow's trends.")
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        predict_target = st.selectbox("Predict for:", ["Gold", "Silver"])
        run_pred = st.button("Run Analysis", type="primary")
        
    if run_pred:
        with st.spinner(f"Analyzing {predict_target} markets..."):
            try:
                target_df = gold_df if predict_target == "Gold" else silver_df
                # Use base price for prediction (no tax), then apply tax to result if needed
                # Actually, simpler to predict on the 'INR_Close' column directly
                
                # Prepare Data
                look_back = 60
                series_to_predict = target_df['INR_Close']
                
                if len(series_to_predict) > 100:
                    x_train, y_train, scaler, raw_scaled = prepare_lstm_data(series_to_predict, lookback=look_back)
                    
                    # Build Model
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dense(units=25))
                    model.add(Dense(units=1))
                    
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
                    
                    # Predict
                    last_60_days = raw_scaled[-look_back:]
                    last_60_days = last_60_days.reshape(1, look_back, 1)
                    
                    pred_scaled = model.predict(last_60_days)
                    pred_price_base = scaler.inverse_transform(pred_scaled)[0][0]
                    
                    # Apply Tax
                    pred_price_final = pred_price_base * tax_multiplier
                    current_price_final = series_to_predict.iloc[-1] * tax_multiplier
                    
                    st.success("Analysis Complete!")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Predicted Price Tomorrow", f"₹{pred_price_final:,.0f}")
                    with m2:
                        diff = pred_price_final - current_price_final
                        st.metric("Expected Change", f"{('▲' if diff>0 else '▼')} ₹{abs(diff):,.0f}")
                        
                else:
                    st.warning("Not enough data history for accurate prediction. Select 'max' period.")
                    
            except Exception as e:
                st.error(f"AI Error: {e}")

with main_tab3:
    st.subheader("Live Market News")
    try:
        # Default to Gold news, or mix? Let's just show Gold for now as primary
        ticker = yf.Ticker(gold_ticker)
        news_list = ticker.news
        
        if news_list:
            for item in news_list:
                with st.container():
                    content = item.get('content', {})
                    title = content.get('title', 'No Title')
                    if 'clickThroughUrl' in content and content['clickThroughUrl']:
                         link = content['clickThroughUrl']['url']
                    elif 'canonicalUrl' in content and content['canonicalUrl']:
                         link = content['canonicalUrl']['url']
                    else:
                         link = "#"
                    
                    summary = content.get('summary', '')
                    provider = content.get('provider', {}).get('displayName', 'Source')
                    
                    st.markdown(f"**[{title}]({link})**")
                    st.caption(f"{provider}")
                    st.markdown(f"_{summary}_")
                    st.markdown("---")
        else:
            st.info("No immediate news headlines found via API.")
    except Exception as e:
        st.error("News service temporarily unavailable.")
