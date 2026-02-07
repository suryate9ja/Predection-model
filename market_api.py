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
# --- RSS CONFIGURATION ---
RSS_FEEDS = [
    {"url": "https://www.metalsdaily.com/news/silver/feed/", "source": "MetalsDaily Silver"},
    {"url": "https://www.metalsdaily.com/news/gold/feed/", "source": "MetalsDaily Gold"},
    {"url": "https://www.kitco.com/rss/category/news/gold.xml", "source": "Kitco Gold"},
    {"url": "https://www.fxstreet.com/rss/news", "source": "FXStreet"},
    # Fallback/Additional sources can be added here
]

def fetch_news():
    """Fetches latest headlines from RSS feeds."""
    logger.info("Fetching news from RSS feeds...")
    try:
        import feedparser
        
        all_news = []
        for feed_cfg in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_cfg["url"])
                for entry in feed.entries[:5]: # Top 5 per feed
                    # Normalize time
                    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
                    if published_parsed:
                        dt = datetime.datetime(*published_parsed[:6])
                        time_ago = dt.strftime("%Y-%m-%d %H:%M")
                    else:
                        time_ago = "Recent"

                    all_news.append({
                        "title": entry.get("title", "No Title"),
                        "link": entry.get("link", "#"),
                        "source": feed_cfg["source"],
                        "time": time_ago,
                        "timestamp": datetime.datetime(*published_parsed[:6]).timestamp() if published_parsed else 0
                    })
            except Exception as e:
                logger.error(f"Error parsing feed {feed_cfg['source']}: {e}")

        # Sort by latest
        all_news.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Keep top 15
        final_news = all_news[:15]
        
        if final_news:
            data_cache["news"] = final_news
            logger.info(f"News update successful. Found {len(final_news)} items.")
        else:
            logger.info("No news items found in RSS feeds.")

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
