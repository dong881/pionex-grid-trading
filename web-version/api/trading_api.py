"""
API module for fetching and analyzing grid trading data.
This module provides functions to be called from the web frontend.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

def fetch_klines_data(symbol="BTC_USDT_PERP", interval="4H", days=14):
    """
    Fetch K-line data from Pionex API
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Time interval (e.g., "4H", "1D")
        days (int): Number of days of historical data
        
    Returns:
        dict: Dictionary containing price data and metadata
    """
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = "https://api.pionex.com/api/v1/market/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 500
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("result", False):
                klines = data["data"]["klines"]
                df = pd.DataFrame(klines, columns=["time", "open", "close", "high", "low", "volume"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].astype(float)
                
                return {
                    "success": True,
                    "data": df.to_dict('records'),
                    "current_price": float(df['close'].iloc[-1]),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "API returned no data",
                    "timestamp": datetime.now().isoformat()
                }
        elif response.status_code == 429:
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "timestamp": datetime.now().isoformat()
            }
        elif response.status_code >= 500:
            return {
                "success": False,
                "error": "API service temporarily unavailable",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout. API may be slow or unavailable.",
            "timestamp": datetime.now().isoformat()
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Network connection error. Please check your internet connection.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def analyze_strategy(df_data, entry_price=None):
    """
    Analyze trading strategy and calculate grid parameters
    
    Args:
        df_data (list): List of price data dictionaries
        entry_price (float): Optional entry price, uses current if not provided
        
    Returns:
        dict: Strategy analysis results
    """
    if not df_data:
        return {"success": False, "error": "No data provided"}
    
    df = pd.DataFrame(df_data)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    current_price = float(df['close'].iloc[-1])
    
    if entry_price is None:
        entry_price = current_price
    
    # Calculate predictions
    predicted_high = float(df['high'].max() * 1.05)
    predicted_low = float(df['low'].min() * 0.95)
    
    # Calculate grid strategy
    price_volatility = (predicted_high - entry_price) / entry_price
    upper_limit = entry_price * (1 + price_volatility * 1.2)
    lower_limit = entry_price * (1 - price_volatility * 0.8)
    
    price_range = upper_limit - lower_limit
    base_grid_count = int(price_range / 100)
    grid_count = max(10, min(200, base_grid_count))
    
    price_ratio = upper_limit / lower_limit
    momentum_factor = 1.8
    raw_leverage = (price_ratio - 1) * momentum_factor
    risk_factor = 0.95
    adjusted_leverage = raw_leverage * risk_factor
    leverage = max(5, min(15, adjusted_leverage))
    
    return {
        "success": True,
        "current_price": current_price,
        "entry_price": entry_price,
        "upper_limit": round(upper_limit, 2),
        "lower_limit": round(lower_limit, 2),
        "price_range": round(price_range, 2),
        "grid_count": grid_count,
        "leverage": round(leverage, 2),
        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2)
    }

def calculate_drawdown(df_data):
    """
    Calculate maximum drawdown from price data
    
    Args:
        df_data (list): List of price data dictionaries
        
    Returns:
        dict: Drawdown analysis
    """
    if not df_data:
        return {"success": False, "error": "No data provided"}
    
    df = pd.DataFrame(df_data)
    df['close'] = df['close'].astype(float)
    
    # Calculate running maximum
    running_max = df['close'].expanding().max()
    drawdown = (df['close'] - running_max) / running_max * 100
    
    max_drawdown = float(drawdown.min())
    max_drawdown_date = df.loc[drawdown.idxmin(), 'time']
    
    # Calculate current drawdown
    current_price = float(df['close'].iloc[-1])
    peak_price = float(running_max.iloc[-1])
    current_drawdown = (current_price - peak_price) / peak_price * 100
    
    return {
        "success": True,
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_date": str(max_drawdown_date),
        "current_drawdown": round(current_drawdown, 2),
        "peak_price": round(peak_price, 2),
        "current_price": round(current_price, 2)
    }

def get_full_analysis(symbol="BTC_USDT_PERP", interval="4H", days=14, entry_price=None):
    """
    Get complete analysis including price data, strategy, and drawdown
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Time interval
        days (int): Number of days of historical data
        entry_price (float): Optional entry price
        
    Returns:
        dict: Complete analysis results
    """
    # Fetch data
    fetch_result = fetch_klines_data(symbol, interval, days)
    
    if not fetch_result["success"]:
        return fetch_result
    
    df_data = fetch_result["data"]
    
    # Analyze strategy
    strategy_result = analyze_strategy(df_data, entry_price)
    
    # Calculate drawdown
    drawdown_result = calculate_drawdown(df_data)
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "price_data": df_data[-30:],  # Last 30 data points for chart
        "strategy": strategy_result,
        "drawdown": drawdown_result
    }

if __name__ == "__main__":
    # Test the API
    result = get_full_analysis()
    print(json.dumps(result, indent=2, default=str))
