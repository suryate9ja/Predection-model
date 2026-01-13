import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold/Silver Price (INR)", layout="wide")

st.title("🇮🇳 Gold & Silver Analytics (INR per 10g/1kg)")
st.markdown("""
<style>
    /* --- FONTS & BASICS --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; 
        color: #E0E0E0;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #161B22; 
        border-right: 1px solid #30363D;
    }
    
    /* --- METRIC CARDS --- */
    div[data-testid="stMetric"] {
        background-color: #1F242D;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #D4AF37; /* Gold Highlight */
    }
    
    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161B22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8B949E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F242D !important;
        color: #D4AF37 !important;
        border-top: 2px solid #D4AF37;
    }

    /* --- CUSTOM HEADERS --- */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* --- TABLES/DATAFRAMES --- */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    /* --- PLOTLY CHART BORDER --- */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪙 Gold & Silver Professional Analytics")
st.caption("Live Spot Prices (USD) converted to INR • Machine Learning Forecasts")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
metal_choice = st.sidebar.selectbox("Select Asset:", ["Gold", "Silver"])
# Added '2y' and 'ytd' to give more options if one fails
period = st.sidebar.selectbox("Data Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

# Ticker Config
metal_ticker = "GC=F" if metal_choice == "Gold" else "SI=F"
currency_ticker = "USDINR=X"

# --- HELPER: FIX DATA STRUCTURE ---
def fix_data_structure(df):
    """Cleans yfinance data: removes multi-index columns and timezones"""
    if df.empty:
        return df
        
    # 1. If columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Remove Timezone information to ensure dates match perfectly
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = df.index.tz_localize(None)
    
    
    return df

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

# --- HELPER: TECHNICAL INDICATORS ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# --- DATA LOADING & CONVERSION FUNCTION ---
@st.cache_data
def load_and_convert_data(metal_sym, curr_sym, period):
    # 1. Fetch Metal Data
    metal_data = yf.download(metal_sym, period=period, progress=False)
    metal_data = fix_data_structure(metal_data)
    
    if metal_data.empty:
        st.error(f"Could not fetch data for {metal_sym}. The market might be closed or the ticker changed.")
        return pd.DataFrame()

    # 2. Fetch Currency Data
    curr_data = yf.download(curr_sym, period=period, progress=False)
    curr_data = fix_data_structure(curr_data)
    
    # 3. Create a common DataFrame based on Metal dates
    df = metal_data.copy()
    
    # 4. Align Currency Data to Metal Dates
    # SMART FILL: ffill() fills forward (Friday rate used for Sat/Sun)
    # bfill() fills backward (if data starts on a holiday, use next day's rate)
    aligned_currency = curr_data['Close'].reindex(df.index).ffill().bfill()
    
    # 5. Conversion Factors
    if metal_choice == "Gold":
        # (Price_USD * USD_INR) / 31.1035 * 10
        factor = 10 / 31.1035
    else:
        # (Price_USD * USD_INR) / 0.0311035
        factor = 1 / 0.0311035

    # 6. Apply Conversion safely
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] * aligned_currency) * factor
        
    # Final cleanup - only drop if genuinely missing
    df = df.dropna()
    df.reset_index(inplace=True)
    return df

try:
    data = load_and_convert_data(metal_ticker, currency_ticker, period)
    
    if data.empty:
        st.warning("Data is currently unavailable for this specific timeframe. Please try selecting '6mo' or '2y'.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📈 Market Dashboard (INR)", "🤖 AI Prediction"])

# ==========================================
# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📊 Market Dashboard", "🤖 AI Forecast Studio"])

# ==========================================
# TAB 1: PREMIMUM DASHBOARD
# ==========================================
with tab1:
    unit = "10g" if metal_choice == "Gold" else "1kg"
    
    # 1. TOP METRICS ROW
    try:
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        day_open = data['Open'].iloc[-1]
        day_high = data['High'].iloc[-1]
        day_low = data['Low'].iloc[-1]
        
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        
        # Volatility (Standard Deviation of last 30 days returns)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.tail(30).std() * np.sqrt(252) * 100 # Annualized
        
        # Layout
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Price", f"₹{current_price:,.0f}", f"{pct_change:.2f}%")
        m2.metric("Day High", f"₹{day_high:,.0f}")
        m3.metric("Day Low", f"₹{day_low:,.0f}")
        m4.metric("30D Volatility", f"{volatility:.1f}%")
        m5.metric("Volume (Est.)", f"{int(data['Volume'].iloc[-1]):,}" if 'Volume' in data and data['Volume'].iloc[-1] > 0 else "N/A")
        
        st.divider()
        
        # 2. ADVANCED CHARTING
        # Calculate Indicators for Chart
        data['SMA_20'] = calculate_sma(data['Close'], 20)
        data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
        
        # Create Subplots: Row 1 = Price, Row 2 = Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{metal_choice} Price Action', 'Volume'),
                            row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(x=data['Date'],
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
        
        # Overlays
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower', showlegend=False), row=1, col=1)
        
        # Volume
        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # Style
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#161B22",
            plot_bgcolor="#0E1117",
            font=dict(color="#E0E0E0"),
            grid=dict(rows=1, columns=1, pattern="independent"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        fig.update_xaxes(showgrid=True, gridcolor='#30363D')
        fig.update_yaxes(showgrid=True, gridcolor='#30363D')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except IndexError:
        st.warning("Not enough data to display metrics.")

# ==========================================
# TAB 2: AI FORECAST STUDIO
# ==========================================
with tab2:
    st.markdown("### 🤖 Predictive Analytics Engine")
    
    if len(data) > 30: 
        # ML Prep
        df_ml = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if df_ml['Volume'].sum() == 0: df_ml.drop('Volume', axis=1, inplace=True)

        # Re-calc indicators efficiently for ML df
        df_ml['SMA_10'] = calculate_sma(df_ml['Close'], window=10)
        df_ml['SMA_20'] = calculate_sma(df_ml['Close'], window=20)
        df_ml['RSI'] = calculate_rsi(df_ml['Close'], window=14)
        df_ml.dropna(inplace=True)

        if len(df_ml) > 20:
             df_ml['Prediction'] = df_ml['Close'].shift(-1)
             
             features = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20', 'RSI']
             if 'Volume' in df_ml.columns: features.append('Volume')
                 
             # Train
             df_clean = df_ml.dropna()
             X = np.array(df_clean[features])
             y = np.array(df_clean['Prediction'])
             
             x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
             rf = RandomForestRegressor(n_estimators=200, random_state=42) # Increased estimators
             rf.fit(x_train, y_train)
             score = rf.score(x_test, y_test)

             # Predict
             last_features = np.array(df_ml[features])[-1:]
             prediction = rf.predict(last_features)[0]
             
             # --- DISPLAY RESULTS ---
             c1, c2 = st.columns([1, 2])
             
             with c1:
                 st.markdown(f"""
                 <div style="background-color: #1F242D; padding: 20px; border-radius: 10px; border: 1px solid #30363D;">
                     <h4 style="color: #8B949E; margin:0;">Model Confidence</h4>
                     <h1 style="font-size: 3rem; color: #D4AF37;">{score:.0%}</h1>
                     <p style="font-size: 0.8rem; color: #8B949E;">R² Score on Test Data</p>
                 </div>
                 """, unsafe_allow_html=True)
                 
             with c2:
                 threshold_buy = current_price * 1.005
                 threshold_sell = current_price * 0.995
                 
                 # Determine Signal
                 if prediction > threshold_buy:
                     signal = "BUY"
                     color = "#2EA043" # Green
                     msg = "Strong upside potential detected."
                 elif prediction < threshold_sell:
                     signal = "SELL"
                     color = "#DA3633" # Red
                     msg = "Downside risk detected."
                 else:
                     signal = "HOLD"
                     color = "#D29922" # Yellow
                     msg = "Market efficiency is high. Low volatility expected."
                
                 st.markdown(f"""
                 <div style="background-color: #1F242D; padding: 20px; border-radius: 10px; border: 1px solid {color};">
                     <div style="display: flex; justify-content: space-between; align-items: center;">
                         <div>
                             <h4 style="color: #8B949E; margin:0;">AI Recommendation</h4>
                             <h1 style="font-size: 3rem; color: {color}; margin: 10px 0;">{signal}</h1>
                             <p style="color: #E0E0E0;">{msg}</p>
                         </div>
                         <div style="text-align: right;">
                             <h4 style="color: #8B949E; margin:0;">Target Price</h4>
                             <h2 style="color: #FFFFFF;">₹{prediction:,.0f}</h2>
                             <p style="color: {color};">{(prediction-current_price):+,.0f} ({(prediction-current_price)/current_price:.2%})</p>
                         </div>
                     </div>
                 </div>
                 """, unsafe_allow_html=True)
             
             # Feature Importance
             st.markdown("### 🧠 Model Logic (Feature Importance)")
             importance = pd.DataFrame({
                 'Feature': features,
                 'Importance': rf.feature_importances_
             }).sort_values(by='Importance', ascending=True)
             
             fig_imp = go.Figure(go.Bar(
                 x=importance['Importance'],
                 y=importance['Feature'],
                 orientation='h',
                 marker_color='#58A6FF'
             ))
             fig_imp.update_layout(
                 height=300,
                 paper_bgcolor="#0E1117",
                 plot_bgcolor="#0E1117",
                 font=dict(color="#E0E0E0"),
                 margin=dict(l=0, r=0, t=0, b=0)
             )
             st.plotly_chart(fig_imp, use_container_width=True)

        else:
             st.warning("Needs more data for indicators.")
    else:
        st.warning("Database too small for AI training.")