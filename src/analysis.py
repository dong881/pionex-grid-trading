import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_last_14_days_klines(symbol="BTC_USDT_PERP", interval="4H"):
    """Fetch the last 14 days of 4-hour K-lines data from the Pionex API."""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=14)).timestamp() * 1000)
    
    url = "https://api.pionex.com/api/v1/market/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 500
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("result", False):
            klines = data["data"]["klines"]
            df = pd.DataFrame(klines, columns=["time", "open", "close", "high", "low", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].astype(float)
            df.set_index("time", inplace=True)
            return df
        else:
            print("Pionex API returned invalid k-line data.")
    else:
        print(f"Failed to fetch K-line data, HTTP status code: {response.status_code}")
    
    return None

def analyze_price_data(df):
    """Analyze the price data to determine optimal entry price and predict future highs and lows."""
    recent_lows = df['low'].rolling(window=6).min()  # Last 24 hours of lows
    optimal_entry_price = recent_lows.iloc[-1]  # Most recent low
    
    # Predictive algorithm for future high and low points (simple linear extrapolation)
    future_high = df['high'].max() * 1.05  # Predicting a 5% increase
    future_low = df['low'].min() * 0.95  # Predicting a 5% decrease
    
    print(f"Optimal Entry Price: {optimal_entry_price:.2f} USD")
    print(f"Predicted High for next 2 weeks: {future_high:.2f} USD")
    print(f"Predicted Low for next 2 weeks: {future_low:.2f} USD")
    
    return optimal_entry_price, future_high, future_low

def analyze_klines(df):
    # Example analysis logic to find suggested entry price and grid limits
    suggested_entry_price = df["close"].mean()  # Replace with your analysis logic
    predicted_low = df["low"].min()  # Replace with your analysis logic
    predicted_high = df["high"].max()  # Replace with your analysis logic
    return suggested_entry_price, predicted_low, predicted_high