import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import time

# --- P1: APP CONFIG & THEME SETUP ---
st.set_page_config(page_title="Gold/Silver Live", layout="wide", page_icon="🪙")

def inject_custom_css(choice):
    if choice == "Gold":
        primary = "#FFD700"  # Gold
        secondary = "#B8860B" 
        bg_card = "#2A2419"
        radial_gradient = "radial-gradient(circle at top, #2C2615 0%, #0E1117 100%)"
    else:
        primary = "#C0C0C0"  # Silver
        secondary = "#A9A9A9" 
        bg_card = "#1F242D"
        radial_gradient = "radial-gradient(circle at top, #1C2028 0%, #0E1117 100%)"
        
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
        }}
        
        .stApp {{
            background: {radial_gradient};
            background-attachment: fixed;
        }}
        
        h1, h2, h3 {{
            color: {primary} !important;
            font-weight: 800;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: rgba(22, 27, 34, 0.6);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
        
        /* METRIC CARDS */
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {bg_card} 0%, rgba(20,20,20,0.8) 100%);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-5px);
            border-color: {primary};
            box-shadow: 0 15px 40px -10px {primary}33; /* 33 is opacity hex */
        }}
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(255,255,255,0.03);
            border-radius: 50px;
            padding: 10px 24px;
            color: #8B949E;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {primary} !important;
            color: #000000 !important;
        }}
        
        /* BUTTONS */
        .stButton button {{
            background-color: transparent;
            border: 1px solid {primary};
            color: {primary};
            transition: 0.3s;
        }}
        .stButton button:hover {{
            background-color: {primary};
            color: black;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- SIDEBAR & THEME SELECTION ---
st.sidebar.title("⚙️ Config")
metal_choice = st.sidebar.radio("Select Asset:", ["Gold", "Silver"], horizontal=True) # Radio looks cleaner for fewer ops
period = st.sidebar.selectbox("Data Period:", ["1y", "2y", "5y", "max"], index=1) 
inject_custom_css(metal_choice)

# Static Tickers
metal_ticker = "GC=F" if metal_choice == "Gold" else "SI=F"
currency_ticker = "USDINR=X"
unit = "10 Grams" if metal_choice == "Gold" else "1 Kilogram"
unit_short = "10g" if metal_choice == "Gold" else "1kg"

# --- HELPER FUNCTIONS ---
def fix_data_structure(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if pd.api.types.is_datetime64_any_dtype(df.index): df.index = df.index.tz_localize(None)
    return df

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_bollinger(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return sma + (std * num_std), sma - (std * num_std)

# --- DATA FETCHING (CACHED 5 MINS) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_data(metal_sym, curr_sym, period):
    # Metal
    metal = yf.download(metal_sym, period=period, progress=False)
    metal = fix_data_structure(metal)
    
    # Currency
    curr = yf.download(curr_sym, period=period, progress=False)
    curr = fix_data_structure(curr)
    
    if metal.empty or curr.empty: return pd.DataFrame()
    
    df = metal.copy()
    aligned_currency = curr['Close'].reindex(df.index).ffill().bfill()
    
    # Factor
    factor = (10 / 31.1035) if metal_choice == "Gold" else (1.0 / 0.0311035)
    
    # Conversion
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] * aligned_currency) * factor
    
    df = df.dropna()
    df.reset_index(inplace=True)
    
    # Add Technicals
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger(df['Close'])
    
    return df

# Main App Logic
try:
    with st.spinner(f"Updating {metal_choice} prices..."):
        data = get_data(metal_ticker, currency_ticker, period)
    
    if data.empty:
        st.error("Market data unavailable. Please try again later.")
        st.stop()
        
    last_update = time.strftime("%H:%M:%S")
    st.sidebar.success(f"Last Updated: {last_update}")
    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()
    
# METRICS
current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2]
change = current_price - prev_price
pct_change = (change / prev_price) * 100
is_up = change >= 0
color_trend = "#00FF00" if is_up else "#FF4B4B" # Bright Green or Bright Red for text

# --- TABS LAYOUT ---
tab_simple, tab_advanced, tab_ai = st.tabs(["🏠 Simple View", "📊 Advanced Analytics", "🔮 AI Forecast"])

# ==========================================
# 1. SIMPLE VIEW (COMMON MAN)
# ==========================================
with tab_simple:
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 0px;'>Current {metal_choice} Rate</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #888;'>Price per {unit} (Includes Currency Conversion)</p>", unsafe_allow_html=True)
    
    # Big Price Display
    html_price = f"""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 5rem; color: #FFFFFF; font-weight: 800; margin: 0;">₹{current_price:,.0f}</h1>
        <h3 style="color: {color_trend}; margin-top: -10px;">
            {'▲' if is_up else '▼'} ₹{abs(change):,.0f} ({pct_change:.2f}%)
            <span style="font-size: 1rem; color: #888; font-weight: 400;">Today</span>
        </h3>
    </div>
    """
    st.markdown(html_price, unsafe_allow_html=True)
    
    st.divider()
    
    # Simple Recommendation Card
    rsi_latest = data['RSI'].iloc[-1]
    
    if rsi_latest > 70:
        advice_title = "Expensive"
        advice_desc = "Ideally wait. Prices are very high right now."
        advice_color = "#FF4B4B" # Red
    elif rsi_latest < 30:
        advice_title = "Good Time to Buy"
        advice_desc = "Prices have fallen significantly. Good value."
        advice_color = "#00FF00" # Green
    else:
        advice_title = "Fair Price"
        advice_desc = "Standard market rate. You can buy if needed."
        advice_color = "#FFA500" # Orange
        
    st.markdown(f"""
    <div style="background-color: rgba(255,255,255,0.05); border-left: 5px solid {advice_color}; padding: 20px; border-radius: 10px;">
        <h3 style="color: {advice_color} !important; margin: 0;">{advice_title}</h3>
        <p style="margin: 5px 0 0 0; font-size: 1.2rem;">{advice_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 24H High/Low visually
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Today's High", f"₹{data['High'].iloc[-1]:,.0f}")
    with c2:
        st.metric("Today's Low", f"₹{data['Low'].iloc[-1]:,.0f}")

# ==========================================
# 2. ADVANCED DATA
# ==========================================
with tab_advanced:
    st.markdown("### Technical Analysis & Data")
    
    # Advanced Plotly Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & Bands', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Price
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='BB Lower', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], line=dict(color='#FFA500', width=1.5), name='SMA 20'), row=1, col=1)

    # Volume
    colors = ['#FF4B4B' if row['Open'] - row['Close'] >= 0 else '#00FF00' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color="#CCC"), margin=dict(l=0,r=0,t=0,b=0))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📄 View Raw Dataset"):
        st.dataframe(data.sort_values(by='Date', ascending=False), height=400, use_container_width=True)

# ==========================================
# 3. AI PREDICTION
# ==========================================
with tab_ai:
    st.markdown("### 🤖 ML Price Forecast")
    
    if len(data) > 50:
        # Prepare Data
        df_ml = data[['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'Volume']].dropna()
        df_ml['Prediction'] = df_ml['Close'].shift(-1)
        
        # Valid Dataset
        df_clean = df_ml.dropna()
        
        # Features and Target
        features = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'RSI']
        X = df_clean[features].values
        y = df_clean['Prediction'].values
        
        # Train
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict Tomorrow
        last_row = df_ml[features].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_row)[0]
        
        confidence = model.score(X, y) # R2 on full set (proxy for fit/confidence)
        
        # Display logic
        threshold = current_price * 1.005
        
        if prediction > threshold:
            sig = "BUY"
            color = "#00FF00"
        elif prediction < current_price * 0.995:
            sig = "SELL"
            color = "#FF4B4B"
        else:
            sig = "HOLD"
            color = "#FFA500"
            
        c1, c2, c3 = st.columns(3)
        with c1:
             st.metric("Predicted Price (T+1)", f"₹{prediction:,.0f}", delta=f"{(prediction-current_price):.0f}")
        with c2:
             st.metric("Model Confidence", f"{confidence:.1%}")
        with c3:
            st.markdown(f"""
            <div style="background-color: {color}22; border: 2px solid {color}; border-radius: 10px; text-align: center; padding: 10px;">
                <h2 style="color: {color} !important; margin:0;">{sig}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("Collecting more data points for accurate prediction...")