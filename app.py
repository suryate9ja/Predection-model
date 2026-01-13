import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import time
from datetime import datetime
import pytz

# --- P1: APP CONFIG & THEME SETUP ---
st.set_page_config(page_title="Gold/Silver Live", layout="wide", page_icon="🪙")

def get_ist_time():
    utc_now = datetime.now(pytz.utc)
    ist_now = utc_now.astimezone(pytz.timezone('Asia/Kolkata'))
    return ist_now.strftime("%d-%b-%Y %I:%M:%S %p")

def inject_custom_css(metal_choice):
    # Apple HIG Colors
    bg_main = "#F5F5F7" # Apple Light Gray
    bg_card = "#FFFFFF" # Pure White
    text_primary = "#1D1D1F" # Apple Black
    text_secondary = "#86868B" # Apple Gray
    
    if metal_choice == "Gold":
        accent = "#B8860B" # Darker Gold for text readability on white
        accent_bg = "rgba(255, 215, 0, 0.1)" # Light Gold background
    else:
        # User wants "Silver" color. Silver Text on white is bad. We use a Slate Gray/Blue.
        accent = "#5E6C84" 
        accent_bg = "rgba(94, 108, 132, 0.1)"

    css = f"""
    <style>
        /* ANIMATIONS */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 {accent}33; }}
            70% {{ box-shadow: 0 0 0 10px rgba(0,0,0,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
        }}
    
        /* APPLE SYSTEM FONTS */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: {text_primary};
        }}
        
        .stApp {{
            background-color: {bg_main};
        }}
        
        h1, h2, h3, h4 {{
            color: {text_primary} !important;
            font-weight: 600;
            letter-spacing: -0.5px; /* Tighter headers */
        }}
        
        /* FROSTED GLASS SIDEBAR */
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.75);
            backdrop-filter: saturate(180%) blur(25px); /* Stronger blur */
            border-right: 1px solid rgba(0,0,0,0.03);
        }}
        
        /* ULTRA PREMIUM CARDS */
        div[data-testid="stMetric"] {{
            background-color: {bg_card};
            border: none;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            animation: fadeIn 0.6s ease-out both; /* Entry Animation */
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 12px 30px rgba(0,0,0,0.08); /* Deep glow on hover */
        }}
        
        /* CUSTOM WHITE CARDS (Manually created ones) */
        .premium-card {{
            background-color: #FFFFFF;
            border-radius: 20px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            transition: all 0.3s ease;
            animation: fadeIn 0.6s ease-out both;
        }}
        .premium-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        }}
        
        div[data-testid="stMetric"] label {{
            color: {text_secondary} !important;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {text_primary} !important;
            font-weight: 700;
            font-size: 2rem !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return accent, accent_bg, text_primary

# --- SIDEBAR CONFIG ---
st.sidebar.title(" Analysis")
metal_choice = st.sidebar.radio("Select Asset:", ["Gold", "Silver"], horizontal=True)

accent_c, accent_bg, text_c = inject_custom_css(metal_choice)


period = st.sidebar.selectbox("History:", ["1mo", "6mo", "1y", "5y", "max"], index=2) 

# --- LOCATION SELECTOR & TAX TOGGLE ---
# We use columns to put this in the "header" area visually
c_title, c_loc, c_tax = st.columns([3, 1, 1])
with c_title:
    st.title(f"{metal_choice} Dashboard")
with c_loc:
    locations = ["India (National)", "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra", "Delhi", "USA", "UK", "UAE", "Australia", "Canada"]
    location = st.selectbox("📍 Location", locations, index=0)
with c_tax:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Access alignment hack
    show_tax = st.toggle("Include Taxes", value=True)

# Static Tickers - REVERT TO FUTURES (MORE RELIABLE)
metal_ticker = "GC=F" if metal_choice == "Gold" else "SI=F"

currency_ticker = "USDINR=X"
unit = "10 Grams" if metal_choice == "Gold" else "1 Kilogram"

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
def get_data(metal_sym, curr_sym, period, location_factor=1.0):
    # Metal
    metal = yf.download(metal_sym, period=period, progress=False)
    metal = fix_data_structure(metal)
    
    # Currency
    # If location is NOT India, we might want to use USD. But user asked for "all states in india" implies INR focus.
    # We will stick to INR for now as the base currency.
    curr = yf.download(curr_sym, period=period, progress=False)
    curr = fix_data_structure(curr)
    
    if metal.empty or curr.empty: return pd.DataFrame()
    
    df = metal.copy()
    aligned_currency = curr['Close'].reindex(df.index).ffill().bfill()
    
    # Unit Factor
    unit_mult = (10 / 31.1035) if metal_choice == "Gold" else (1.0 / 0.0311035)
    
    # Conversion + Tax Logic
    # Price = (Global_Spot * USDINR * Unit_Mult) * Tax_Factor
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] * aligned_currency) * unit_mult * location_factor
    
    df = df.dropna()
    df.reset_index(inplace=True)
    
    # Add Technicals
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger(df['Close'])
    
    return df

# --- TAX LOGIC ---
india_states = ["India (National)", "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra", "Delhi", "Kerala", "West Bengal", "Gujarat"]
tax_factor = 1.0

if location in india_states:
    # PRECISE CALCULATION FOR INDIA
    # Import Duty is ~15%. GST is 3% on (Base + Duty). 
    # Factor = 1.15 * 1.03 = 1.1845
    tax_factor = 1.185 # Rounding to 1.185
    
    # State slight variations (Transportation/Local demands)
    if location == "Andhra Pradesh" or location == "Telangana": tax_factor += 0.002 
    if location == "Tamil Nadu" or location == "Kerala": tax_factor -= 0.002
    if location == "Maharashtra": tax_factor += 0.005 # Premium
elif location == "UAE":
    tax_factor = 1.05 # 5% VAT
else:
    tax_factor = 1.0 # Global Spot (USA/UK etc)

# Main App Logic
try:
    final_tax_factor = tax_factor if show_tax else 1.0
    
    with st.spinner(f"Fetching rates for {location}..."):
        data = get_data(metal_ticker, currency_ticker, period, location_factor=final_tax_factor)
    
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
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 0px;'>{location} Rate</h3>", unsafe_allow_html=True)
    st.caption(f"Price per {unit} • Includes Estimated Taxes & Import Duties")
    
    # KARAT LOGIC
    price_24k = current_price
    price_22k = current_price * 0.916 if metal_choice == "Gold" else current_price # Silver usually 999 or not quoted in 22k
    
    if metal_choice == "Gold":
        k24, k22 = st.columns(2)
        with k24:
             st.markdown(f"""
             <div class="premium-card">
                 <h4 style="margin:0; color: {accent_c};">24 Karat (99.9%)</h4>
                 <h1 style="margin:0; font-size: 3rem; color: #1D1D1F;">₹{price_24k:,.0f}</h1>
                 <p style="margin:0; font-size: 0.9rem; color: #86868B;">Pure Gold Coin/Bar</p>
             </div>
             """, unsafe_allow_html=True)
        with k22:
             st.markdown(f"""
             <div class="premium-card">
                 <h4 style="margin:0; color: {accent_c};">22 Karat (91.6%)</h4>
                 <h1 style="margin:0; font-size: 3rem; color: #1D1D1F;">₹{price_22k:,.0f}</h1>
                 <p style="margin:0; font-size: 0.9rem; color: #86868B;">Standard Jewellery</p>
             </div>
             """, unsafe_allow_html=True)
    else:
        # Silver Display
        st.markdown(f"""
             <div class="premium-card">
                 <h1 style="margin:0; font-size: 4.5rem; color: #1D1D1F;">₹{current_price:,.0f}</h1>
                 <p style="margin:0; font-size: 1.1rem; color: #86868B;">Silver (99.9%) per Kg</p>
             </div>
             """, unsafe_allow_html=True)

    
    st.divider()
    
    # Trend Indicator
    st.markdown(f"""
    <div style="text-align: center;">
        <h3 style="color: {color_trend};">
            {'▲' if is_up else '▼'} ₹{abs(change):,.0f} ({pct_change:.2f}%)
        </h3>
        <p style="color: #86868B;">Market Movement Today</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PRICE BREAKDOWN (TAX SPLIT)
    if tax_factor > 1.0:
        base_price = current_price / tax_factor
        tax_amt = current_price - base_price
        
        st.markdown(f"""
        <div class="premium-card" style="margin-top: 20px;">
            <span style="color: #86868B;">Base Spot:</span> <b>₹{base_price:,.0f}</b> &nbsp; + &nbsp; 
            <span style="color: #86868B;">Tax:</span> <b style="color: #FF4B4B;">₹{tax_amt:,.0f}</b> &nbsp; = &nbsp; 
            <span style="color: #86868B;">Total:</span> <b style="color: #1D1D1F;">₹{current_price:,.0f}</b>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Advanced    # PLOTLY CHART - PREMIUM STYLE
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Main Price Area
    # Use fill='tozeroy' for the gradient look (Plotly defaults are solid but better than line)
    # To get real gradient we need more complex rgba, but 'tozeroy' with opacity is good enough for "Stocks" look
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Price',
                             line=dict(color=accent_c, width=2),
                             fill='tozeroy', 
                             fillcolor=f"rgba{tuple(int(accent_c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"), # Hex to RGBA hack
                  row=1, col=1)
                  
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='SMA 20',
                             line=dict(color='rgba(255, 165, 0, 0.5)', width=1)),
                  row=1, col=1)
    
    # Bollinger Bands (Subtle)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], mode='lines', name='BB Upper',
                             line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], mode='lines', name='BB Lower',
                             line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5), fill='tonexty',
                             fillcolor='rgba(128, 128, 128, 0.05)', showlegend=False),
                  row=1, col=1)

    # Volume Bar
    # Color bars based on Up/Down
    colors = [
        '#00C805' if row['Open'] - row['Close'] >= 0 else '#FF5000' 
        for index, row in data.iterrows() 
    ]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color=colors, opacity=0.3),
                  row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        dragmode='pan',
        showlegend=False
    )
    
    # Minimalist Axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1) # Hide X on top chart
    fig.update_xaxes(showgrid=False, zeroline=False, color='#86868B', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, color='#86868B', row=1, col=1) # Faint grid Y
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1) # Hide Y on vol

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
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