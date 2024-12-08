from fetch_data import fetch_klines
from analysis import analyze_klines
from strategy import calculate_grid_strategy

def validate_predictions(predicted_low, predicted_high):
    """Validate prediction values are reasonable"""
    if predicted_low >= predicted_high:
        return False
    if predicted_low <= 0:
        return False
    return True

def calculate_safe_leverage(predicted_low, predicted_high):
    """Calculate leverage with more aggressive bounds (5-15x)"""
    raw_leverage = (predicted_high - predicted_low) / predicted_low * 1.5  # Added momentum multiplier
    if raw_leverage < 5:
        return 5
    elif raw_leverage > 15:
        return 15
    return raw_leverage

def calculate_grid_count(price_range):
    """Calculate grid count between 10-200"""
    # Base calculation on price range
    suggested_count = int(price_range / 100)  # One grid per $100 range
    if suggested_count < 10:
        return 10
    elif suggested_count > 200:
        return 200
    return suggested_count

def main():
    symbol = "BTC_USDT_PERP"
    interval = "4H"
    limit = 14 * 6  # 14 days of 4-hour intervals
    df = fetch_klines(symbol, interval, limit)
    
    if df is not None:
        suggested_entry_price, predicted_low, predicted_high = analyze_klines(df)
        
        if not validate_predictions(predicted_low, predicted_high):
            print("Error: Invalid prediction values")
            return
            
        price_range = predicted_high - predicted_low
        leverage = calculate_safe_leverage(predicted_low, predicted_high)
        grid_count = calculate_grid_count(price_range)
        
        print("\nGrid Trading Strategy:")
        print(f"Entry Price: {suggested_entry_price:.2f} USD")
        print(f"Lower Limit: {predicted_low:.2f} USD")
        print(f"Upper Limit: {predicted_high:.2f} USD")
        print(f"Price Range: {price_range:.2f} USD")
        print(f"Number of Grids: {grid_count}")
        print(f"Leverage: {leverage:.2f}x")

if __name__ == "__main__":
    main()