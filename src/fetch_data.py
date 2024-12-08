import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_klines(symbol, interval, limit):
    url = "https://api.pionex.com/api/v1/market/klines"
    params = {
        "symbol": symbol, 
        "interval": interval, 
        "limit": 1  # Get only most recent data
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
    return None

def fetch_last_14_days_klines(symbol="BTC_USDT_PERP", interval="4H"):
    """Fetch the last 14 days of 4-hour K-lines data from the Pionex API."""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=14)).timestamp() * 1000)
    
    url = "https://api.pionex.com/api/v1/market/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 500,
        "startTime": start_time,
        "endTime": end_time
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

def print_fetched_data(df):
    """Print the fetched K-lines data for verification."""
    if df is not None:
        print("Fetched K-lines Data:")
        print(df)
    else:
        print("No data to display.")

# Example usage
if __name__ == "__main__":
    df = fetch_last_14_days_klines()
    print_fetched_data(df)