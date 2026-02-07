from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import datetime
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GOLD_TICKER = "GC=F"
SILVER_TICKER = "SI=F"
CURRENCY_TICKER = "USDINR=X"
NEWS_URL = "https://www.google.com/search?q=gold+silver+price+news+india&tbm=nws"

# --- GLOBAL DATA CACHE ---
# This acts as our in-memory database, updated every 15 minutes.
data_cache: Dict[str, Any] = {
    "last_updated": None,
    "prices": {},
    "news": []
}

# --- DATA FETCHING FUNCTIONS ---
def fetch_prices():
    """Fetches real-time Gold & Silver prices (converted to INR) using yfinance."""
    logger.info("Fetching market prices...")
    try:
        # Fetch data for Gold, Silver, and USD/INR
        tickers = f"{GOLD_TICKER} {SILVER_TICKER} {CURRENCY_TICKER}"
        data = yf.download(tickers, period="1d", interval="1m", progress=False, group_by="ticker")
        
        if data.empty:
            logger.warning("No data received from yfinance.")
            return

        # Extract latest Close prices
        # Note: yfinance structure varies depending on group_by. 
        # With group_by='ticker', it returns a MultiIndex.
        
        # Helper to get latest valid value
        def get_latest(ticker_data):
            if ticker_data.empty: return 0.0
            return ticker_data['Close'].dropna().iloc[-1]

        price_gold_usd = get_latest(data[GOLD_TICKER])
        price_silver_usd = get_latest(data[SILVER_TICKER])
        price_forex = get_latest(data[CURRENCY_TICKER])

        # Conversion Factors
        # Gold: Troy Oz -> 10 Grams
        gold_factor = 10 / 31.1035
        # Silver: Troy Oz -> 1 Kg
        silver_factor = 1 / 0.0311035

        # Calculate INR Prices (Base)
        gold_inr_base = price_gold_usd * price_forex * gold_factor
        silver_inr_base = price_silver_usd * price_forex * silver_factor

        # Add Tax (3% GST) - Standard Indian Market Practice
        gold_inr_tax = gold_inr_base * 1.03
        silver_inr_tax = silver_inr_base * 1.03

        data_cache["prices"] = {
            "gold": {
                "base_inr_10g": round(gold_inr_base, 2),
                "tax_inr_10g": round(gold_inr_tax, 2),
                "global_usd_oz": round(price_gold_usd, 2)
            },
            "silver": {
                "base_inr_1kg": round(silver_inr_base, 2),
                "tax_inr_1kg": round(silver_inr_tax, 2),
                "global_usd_oz": round(price_silver_usd, 2)
            },
            "forex_usd_inr": round(price_forex, 4)
        }
        data_cache["last_updated"] = datetime.datetime.now().isoformat()
        logger.info("Price update successful.")

    except Exception as e:
        logger.error(f"Error fetching prices: {e}")

def fetch_news():
    """Scrapes latest headlines from Google News (HTML) to get generic market sentiment."""
    logger.info("Fetching news headlines...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(NEWS_URL, headers=headers)
        if response.status_code != 200:
            logger.warning(f"News fetch failed with status {response.status_code}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        search_results = soup.select('div.SoaBEf') # Google News card selector (may change)
        
        news_items = []
        for result in search_results[:5]: # Top 5 news
            title_div = result.select_one('div.MBeuO')
            if title_div:
                title = title_div.get_text()
                link = result.find('a')['href'] if result.find('a') else "#"
                source_div = result.select_one('div.NUnG9d span')
                source = source_div.get_text() if source_div else "Unknown"
                time_div = result.select_one('div.OSrXXb span')
                time_ago = time_div.get_text() if time_div else ""
                
                news_items.append({
                    "title": title,
                    "link": link,
                    "source": source,
                    "time": time_ago
                })
        
        if news_items:
            data_cache["news"] = news_items
            logger.info(f"News update successful. Found {len(news_items)} items.")
        else:
            logger.info("No news items parsed (selectors might iterate).")

    except Exception as e:
        logger.error(f"Error fetching news: {e}")

def update_all_data():
    fetch_prices()
    fetch_news()

# --- LIFECYCLE INTERFACE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Init scheduler and perform first fetch
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_all_data, 'interval', minutes=15)
    scheduler.start()
    
    # Run an immediate update so we have data on boot
    logger.info("Performing initial data fetch...")
    update_all_data()
    
    yield
    
    # Shutdown
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan, title="Gold & Silver Market API")

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Gold & Silver Market API", "version": "1.0"}

@app.get("/latest")
def get_latest_data():
    """Returns the latest cached market data and news."""
    if not data_cache["last_updated"]:
        # Try a quick fetch if cache is empty (e.g. startup race condition)
        update_all_data()

    if not data_cache["last_updated"]:
        raise HTTPException(status_code=503, detail="Data not yet available. Please try again in 1 minute.")
    
    return data_cache

@app.post("/refresh")
def force_refresh():
    """Manually triggers a data refresh."""
    update_all_data()
    return {"status": "Refresh triggered", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
