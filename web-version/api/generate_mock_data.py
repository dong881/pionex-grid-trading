"""
Mock data generator for testing when API is unavailable.
Generates realistic-looking data for the web interface.
"""
import json
import os
from datetime import datetime, timedelta
import random

def generate_mock_price_data(days=14, base_price=95000):
    """Generate mock price data"""
    data = []
    current_time = datetime.now()
    
    # Generate 4-hour intervals for the specified days
    num_intervals = days * 6  # 6 intervals per day (4 hours each)
    
    for i in range(num_intervals):
        timestamp = current_time - timedelta(hours=4 * (num_intervals - i - 1))
        
        # Add some randomness to price movement
        volatility = 0.02  # 2% volatility
        price_change = random.uniform(-volatility, volatility)
        current_price = base_price * (1 + price_change)
        
        # Generate OHLCV data
        high = current_price * (1 + random.uniform(0, 0.01))
        low = current_price * (1 - random.uniform(0, 0.01))
        open_price = base_price
        close_price = current_price
        volume = random.uniform(1000, 5000)
        
        data.append({
            "time": timestamp.isoformat(),
            "open": round(open_price, 2),
            "close": round(close_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "volume": round(volume, 2)
        })
        
        # Update base price for next interval
        base_price = current_price
    
    return data

def generate_mock_analysis(days=14):
    """Generate complete mock analysis"""
    price_data = generate_mock_price_data(days=days)
    
    # Calculate stats from mock data
    current_price = price_data[-1]["close"]
    all_highs = [d["high"] for d in price_data]
    all_lows = [d["low"] for d in price_data]
    
    predicted_high = max(all_highs) * 1.05
    predicted_low = min(all_lows) * 0.95
    
    # Calculate strategy parameters
    entry_price = current_price
    price_volatility = (predicted_high - entry_price) / entry_price
    upper_limit = entry_price * (1 + price_volatility * 1.2)
    lower_limit = entry_price * (1 - price_volatility * 0.8)
    price_range = upper_limit - lower_limit
    
    grid_count = max(10, min(200, int(price_range / 100)))
    
    price_ratio = upper_limit / lower_limit
    leverage = max(5, min(15, (price_ratio - 1) * 1.8 * 0.95))
    
    # Calculate drawdown
    prices = [d["close"] for d in price_data]
    max_price = max(prices)
    min_price = min(prices)
    max_drawdown = ((min_price - max_price) / max_price) * 100
    current_drawdown = ((current_price - max_price) / max_price) * 100
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "symbol": "BTC_USDT_PERP",
        "interval": "4H",
        "days": days,
        "price_data": price_data[-30:],  # Last 30 data points
        "strategy": {
            "success": True,
            "current_price": round(current_price, 2),
            "entry_price": round(entry_price, 2),
            "upper_limit": round(upper_limit, 2),
            "lower_limit": round(lower_limit, 2),
            "price_range": round(price_range, 2),
            "grid_count": grid_count,
            "leverage": round(leverage, 2),
            "predicted_high": round(predicted_high, 2),
            "predicted_low": round(predicted_low, 2)
        },
        "drawdown": {
            "success": True,
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_date": price_data[prices.index(min_price)]["time"],
            "current_drawdown": round(current_drawdown, 2),
            "peak_price": round(max_price, 2),
            "current_price": round(current_price, 2)
        }
    }

def generate_mock_data_files():
    """Generate mock data files"""
    output_dir = "../static/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for different periods
    periods = {"7d": 7, "14d": 14, "30d": 30}
    
    for period, days in periods.items():
        data = generate_mock_analysis(days=days)
        output_file = os.path.join(output_dir, f"analysis_{period}.json")
        with open(output_file, 'w') as f:
            json.dump(data, indent=2, fp=f)
        print(f"Generated mock data: {output_file}")
    
    print("\nMock data generation complete!")

if __name__ == "__main__":
    generate_mock_data_files()
