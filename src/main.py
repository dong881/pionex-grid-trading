from fetch_data import fetch_klines
from analysis import analyze_klines
from strategy import calculate_grid_strategy

def main():
    # Fetch the latest 14 days of 4-hour K-lines data
    symbol = "BTC_USDT_PERP"
    interval = "4H"
    limit = 14 * 6  # 14 days of 4-hour intervals
    df = fetch_klines(symbol, interval, limit)
    
    if df is not None:
        print("Fetched K-lines data:")
        print(df)
        
        # Analyze the K-lines data to find the suggested entry price and grid limits
        suggested_entry_price, predicted_low, predicted_high = analyze_klines(df)
        
        # Calculate the grid strategy
        grid_count, leverage = calculate_grid_strategy(predicted_low, predicted_high)
        
        print(f"Suggested Entry Price: {suggested_entry_price:.2f} USD")
        print(f"Grid Trading Suggestion:")
        print(f"  Upper Price Limit: {predicted_high:.2f} USD")
        print(f"  Lower Price Limit: {predicted_low:.2f} USD")
        print(f"  Grid Count: {grid_count}")
        print(f"  Suggested Leverage: {leverage:.2f}x")
    else:
        print("Failed to fetch K-lines data.")

if __name__ == "__main__":
    main()