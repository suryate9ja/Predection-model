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

@st.cache_data(ttl=900, show_spinner=False) # Cache news for 15 mins
def fetch_news(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        return t.news
    except:
        return []

def inject_custom_css(metal_choice):
    # Ultra Premium Dark Colors
    bg_main = "radial-gradient(circle at 50% 0%, #1a202c 0%, #0d1117 100%)" # Deep Blue-Grey to Black
    bg_card = "rgba(255, 255, 255, 0.05)" # Glass Dark
    text_primary = "#FFFFFF"
    text_secondary = "#94a3b8" # Slate 400
    
    if metal_choice == "Gold":
        accent = "#FFD700" # Neon Gold
        accent_bg = "rgba(255, 215, 0, 0.15)"
        grad_text = "linear-gradient(45deg, #FFD700, #FDB931)"
    else:
        accent = "#E0E0E0" # Metallic White/Silver
        accent_bg = "rgba(220, 220, 220, 0.1)" 
        grad_text = "linear-gradient(45deg, #FFFFFF, #B0B0B0)" # White to Silver Gradient

    css = f"""
    <style>
        /* IMPORT OUTFIT FONT */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* ANIMATIONS */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes glow {{
            0% {{ box-shadow: 0 0 5px {accent}20; }}
            50% {{ box-shadow: 0 0 20px {accent}60, 0 0 10px {accent}; }}
            100% {{ box-shadow: 0 0 5px {accent}20; }}
        }}
    
        /* GLOBAL STYLES */
        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
            color: {text_primary};
        }}
        
        .stApp {{
            background: {bg_main};
            background-attachment: fixed;
        }}
        
        h1, h2, h3, h4 {{
            color: {text_primary} !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        
        /* FROSTED GLASS SIDEBAR */
        [data-testid="stSidebar"] {{
            background-color: rgba(13, 17, 23, 0.8);
            backdrop-filter: saturate(180%) blur(20px);
            border-right: 1px solid rgba(255,255,255,0.05);
        }}
        
        /* ULTRA PREMIUM DARK CARDS */
        div[data-testid="stMetric"], .premium-card {{
            background: {bg_card};
            border: 1px solid rgba(255,255,255,0.08); /* Subtle border */
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5); /* Deep shadow */
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy */
            animation: fadeIn 0.8s ease-out both;
            backdrop-filter: blur(10px);
        }}
        
        div[data-testid="stMetric"]:hover, .premium-card:hover {{
            transform: translateY(-5px) scale(1.02);
            border-color: {accent};
            box-shadow: 0 0 30px {accent}33; /* Glow effect */
        }}
        
        /* Typography adjust inside cards */
        div[data-testid="stMetric"] label {{
            color: {text_secondary} !important;
            font-size: 0.85rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {text_primary} !important;
            text-shadow: 0 0 20px rgba(0,0,0,0.5);
        }}
        
        /* TABS - NEON STYLE */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 15px;
            background-color: transparent; 
            padding: 5px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 8px 20px;
            color: {text_secondary};
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent}22 !important; /* Low opacity accent BG */
            border-color: {accent} !important;
            color: {accent} !important;
            box-shadow: 0 0 15px {accent}40;
        }}
    
        /* GRADIENT TEXT HELPER */
        .grad-text {{
            background: {grad_text};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }}
        
        /* CHECKBOX */
        [data-baseweb="checkbox"] div {{
            background-color: {accent} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return accent, accent_bg, text_primary, grad_text

# --- SIDEBAR CONFIG ---
st.sidebar.title("Data Analytics")
metal_choice = st.sidebar.radio("Select Asset:", ["Gold", "Silver"], horizontal=True)

accent_c, accent_bg, text_c, grad_txt = inject_custom_css(metal_choice)


period = st.sidebar.selectbox("History:", ["1mo","3mo", "6mo", "1y", "5y", "max"], index=2) 

# --- LOCATION SELECTOR & TAX TOGGLE ---
# We use columns to put this in the "header" area visually
c_title, c_loc, c_tax = st.columns([3, 1, 1])
with c_title:
    st.title(f"{metal_choice} Dashboard")
with c_loc:
    locations = ["India (National)", "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra", "Delhi", "USA", "UK", "UAE", "Australia", "Canada"]
    location = st.selectbox("📍 Location", locations, index=0)
with c_tax:
    # Improved Alignment: Use a container with padding to push the toggle down cleanly
    st.markdown("""
        <style>
        .stToggle {
            margin-top: 15px; /* Adjust this to align with the Selectbox */
        }
        </style>
    """, unsafe_allow_html=True)
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
tab_simple, tab_advanced, tab_ai, tab_news = st.tabs(["🏠 Simple View", "📊 Advanced Analytics", "🔮 AI Forecast", "📰 Live Updates"])

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
                 <h4 style="margin:0; color: {accent_c}; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">24 Karat (99.9%)</h4>
                 <h1 style="margin:0; font-size: 3rem; color: #FFF;">₹{price_24k:,.0f}</h1>
                 <p style="margin:0; font-size: 0.9rem; color: {text_c}; opacity: 0.7;">Pure Gold</p>
             </div>
             """, unsafe_allow_html=True)
        with k22:
             st.markdown(f"""
             <div class="premium-card">
                 <h4 style="margin:0; color: {accent_c}; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">22 Karat (91.6%)</h4>
                 <h1 style="margin:0; font-size: 3rem; color: #FFF;">₹{price_22k:,.0f}</h1>
                 <p style="margin:0; font-size: 0.9rem; color: {text_c}; opacity: 0.7;">Jewellery Standard</p>
             </div>
             """, unsafe_allow_html=True)
    else:
        # Silver Display
        st.markdown(f"""
             <div class="premium-card">
                 <h1 class="grad-text" style="margin:0; font-size: 4.5rem;">₹{current_price:,.0f}</h1>
                 <p style="margin:0; font-size: 1.1rem; color: {text_c}; opacity: 0.8;">Silver (99.9%) per Kg</p>
             </div>
             """, unsafe_allow_html=True)

    
    st.divider()
    
    # Big Price Display - LUXURY STYLE
    html_price = f"""
    <div style="text-align: center; padding: 20px;">
        <h1 class="grad-text" style="font-size: 6rem; letter-spacing: -2px; margin: 0; padding-bottom: 10px;">₹{current_price:,.0f}</h1>
        <h3 style="color: {color_trend}; margin-top: -10px; font-weight: 600;">
            {'▲' if is_up else '▼'} ₹{abs(change):,.0f} ({pct_change:.2f}%)
            <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">LIVE</span>
        </h3>
    </div>
    """
    st.markdown(html_price, unsafe_allow_html=True)
    
    st.divider()
    
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

    # Gradient Area Chart
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Price',
                             line=dict(color=accent_c, width=3), # Thicker line
                             fill='tozeroy', 
                             fillcolor=f"rgba{tuple(int(accent_c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"),
                  row=1, col=1)
                  
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='SMA 20',
                             line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot')), # Dashed white SMA
                  row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], mode='lines', 
                             line=dict(color='rgba(255,255,255,0.1)', width=0.5), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], mode='lines', 
                             line=dict(color='rgba(255,255,255,0.1)', width=0.5), fill='tonexty',
                             fillcolor='rgba(255,255,255,0.02)', showlegend=False),
                  row=1, col=1)

    # Volume Bar
    colors = ['#00FF00' if row['Open'] - row['Close'] >= 0 else '#FF0000' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color=colors, opacity=0.5),
                  row=2, col=1)

    fig.update_layout(
        template='plotly_dark', # DARK THEME
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        # paper_bgcolor='rgba(0,0,0,0)', # Removed as per instruction
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        dragmode='pan',
        showlegend=False,
        font=dict(family="Outfit, sans-serif")
    )
    
    # Axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, color='#94a3b8', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#94a3b8', row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)

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
        
        if prediction > current_price:
            sig = "BUY"
            color = "#00FF00"
            st.success(f"📈 AI Signal: BULLISH (Target: ₹{prediction:,.0f})")
        else:
            sig = "SELL"
            color = "#FF4B4B"
            st.error(f"📉 AI Signal: BEARISH (Target: ₹{prediction:,.0f})")
            
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

# ==========================================
# 4. LIVE NEWS
# ==========================================
with tab_news:
    st.markdown("### 🌍 Global Market Intelligence")
    st.caption("Real-time updates from reliable financial sources via Yahoo Finance.")
    
    # 1. AI Summary Snippet (Sentiment)
    # We can infer sentiment from RSI for a "Quick Glance"
    rsi_val = list(data['RSI'])[-1] if 'RSI' in data and len(data) > 0 else 50
    sentiment = "BULLISH" if rsi_val > 55 else ("BEARISH" if rsi_val < 45 else "NEUTRAL")
    sent_color = "#00FF00" if sentiment == "BULLISH" else ("#FF4B4B" if sentiment == "BEARISH" else "#CCCCCC")
    
    st.markdown(f"""
    <div class="premium-card" style="margin-bottom: 20px; border-left: 4px solid {sent_color};">
        <h4 style="margin:0; color: {accent_c};">🤖 AI Analyst Note</h4>
        <p style="margin:5px 0 0 0; color: {text_c}; font-size: 0.95rem;">
            Current technical indicators (RSI: {rsi_val:.1f}) suggest a 
            <b style="color: {sent_color};">{sentiment}</b> sentiment for {metal_choice}.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. News Feed
    news_items = fetch_news(metal_ticker)
    
    if not news_items:
        st.info("No recent news found. Check back later.")
    else:
        for item in news_items:
            try:
                content = item.get('content', {})
                title = content.get('title', 'Market Update')
                # YFinance structure varies, try top level then content
                if title == 'Market Update': title = item.get('title', 'Market Update')
                
                link = content.get('clickThroughUrl', {}).get('url', '#')
                if link == '#': link = item.get('link', '#')
                
                pub_str = content.get('pubDate', '')
                
                # Thumbnail
                thumb_url = None
                thumb_data = content.get('thumbnail', {})
                if thumb_data:
                    thumb_url = thumb_data.get('originalUrl', None)
                
                # News Card
                with st.container():
                    c_thumb, c_text = st.columns([1, 4])
                    
                    with c_thumb:
                        if thumb_url:
                            st.image(thumb_url, use_container_width=True)
                        else:
                            st.markdown(f"<div style='height: 80px; background: {accent_bg}; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 2rem;'>📰</div>", unsafe_allow_html=True)
                            
                    with c_text:
                        st.markdown(f"""
                        <div style="padding-left: 10px;">
                            <a href="{link}" target="_blank" style="text-decoration: none;">
                                <h4 style="margin: 0; color: #FFF; font-size: 1.1rem; font-weight: 600;">{title}</h4>
                            </a>
                            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                                {pub_str[:10] if pub_str else 'Today'} • <a href="{link}" style="color: {accent_c}; text-decoration: none;">Read Full Story ↗</a>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()
            except Exception as e:
                continue