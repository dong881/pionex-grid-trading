from fetch_data import fetch_klines
from analysis import analyze_klines
from strategy import calculate_grid_strategy, validate_grid_parameters

def validate_predictions(predicted_low, predicted_high):
    """Validate prediction values are reasonable"""
    if predicted_low >= predicted_high:
        return False
    if predicted_low <= 0:
        return False
    return True

def main():
    symbol = "BTC_USDT_PERP"
    interval = "4H"
    limit = 14 * 6  # 14 days of 4-hour intervals
    df = fetch_klines(symbol, interval, limit)
    
    if df is not None:
        current_price = df['close'].iloc[-1]  # Get the latest price
        print(f"\nCurrent Price: {current_price:.2f} USD")
        
        # Get user input for desired entry price
        try:
            entry_price_input = input("Enter your desired entry price (or press Enter to use current price): ")
            if entry_price_input.strip() == "":
                entry_price = round(current_price, -2)  # Use current price rounded to nearest hundred
                print(f"Using current price as entry price: {entry_price:.2f} USD")
            else:
                entry_price = float(entry_price_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
        
        suggested_entry_price, predicted_low, predicted_high = analyze_klines(df)
        
        if not validate_predictions(predicted_low, predicted_high):
            print("Error: Invalid prediction values")
            return
            
        grid_count, leverage, upper_limit, lower_limit = calculate_grid_strategy(entry_price, predicted_high)
        
        if not validate_grid_parameters(grid_count, leverage, upper_limit, lower_limit):
            print("Error: Invalid grid parameters calculated")
            return
        
        print("\nGrid Trading Strategy (Aggressive):")
        print(f"Entry Price: {entry_price:.2f} USD")
        print(f"Upper Limit: {upper_limit:.2f} USD")
        print(f"Lower Limit: {lower_limit:.2f} USD")
        print(f"Price Range: {(upper_limit - lower_limit):.2f} USD")
        print(f"Number of Grids: {grid_count}")
        print(f"Leverage: {leverage:.2f}x")

if __name__ == "__main__":
    main()