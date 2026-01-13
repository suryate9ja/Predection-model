import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold/Silver Price (INR)", layout="wide")

st.title("🇮🇳 Gold & Silver Analytics (INR per 10g/1kg)")
st.markdown("""
This dashboard converts global Spot Prices (USD/oz) to **Indian Rupees (INR)**.
* **Gold:** Price per **10 Grams**
* **Silver:** Price per **1 Kilogram**
""")

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
# TAB 1: DASHBOARD
# ==========================================
with tab1:
    unit = "10g" if metal_choice == "Gold" else "1kg"
    st.subheader(f"{metal_choice} Price History (₹ per {unit})")
    
    # Calculate Metrics
    try:
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric(label=f"Current Price (₹/{unit})", value=f"₹{current_price:,.0f}", delta=f"{pct_change:.2f}%")
        col2.metric(label="Previous Close", value=f"₹{prev_close:,.0f}")
        col3.metric(label="Data Source", value="Global Spot + FX Rate")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'],
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name='Price in INR'))
        
        # FIXED: Updated for 2026/Streamlit new standards
        fig.update_layout(title=f'{metal_choice} Price in INR', yaxis_title=f'Price (₹)', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch") # 'width="stretch"' makes it responsive
    except IndexError:
        st.warning("Not enough data to display metrics.")

# ==========================================
# TAB 2: AI PREDICTION
# ==========================================
with tab2:
    st.subheader(f"🔮 AI Prediction (Next Day in INR)")
    
    if len(data) > 20: 
        # ML Prep
        df_ml = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Handle zero volume (common in forex/calculated data)
        if df_ml['Volume'].sum() == 0:
             df_ml.drop('Volume', axis=1, inplace=True)

        df_ml['Prediction'] = df_ml['Close'].shift(-1)
        
        X = np.array(df_ml.drop(['Prediction'], axis=1))[:-1]
        y = np.array(df_ml['Prediction'])[:-1]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(x_train, y_train)
        
        score = rf.score(x_test, y_test)
        st.info(f"Model Accuracy (R² Score): **{score:.2%}**")

        # Predict
        last_day_data = np.array(df_ml.drop(['Prediction'], axis=1))[-1:]
        prediction = rf.predict(last_day_data)[0]
        
        st.divider()
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            st.write(f"### Predicted Price Tomorrow:")
            st.write(f"# ₹{prediction:,.0f}")
        
        with col_pred2:
            threshold = current_price * 1.005
            st.write("### AI Recommendation:")
            if prediction > threshold:
                st.success("🟢 **BUY SIGNAL**")
                st.write("Predicted to rise significantly.")
            elif prediction < current_price:
                st.error("🔴 **SELL/WAIT SIGNAL**")
                st.write("Predicted to fall.")
            else:
                st.warning("🟡 **HOLD**")
                st.write("Market expected to remain flat.")
    else:
        st.warning("Not enough data points to train the AI model. Please select a longer time period.")