import requests
import pandas as pd

def set_grid_limits(predicted_low, predicted_high, grid_count):
    """Set upper and lower limits for grid trading."""
    grid_size = (predicted_high - predicted_low) / grid_count
    upper_limit = predicted_high
    lower_limit = predicted_low
    return upper_limit, lower_limit, grid_size

def calculate_take_profit_and_stop_loss(entry_price, grid_size):
    """Establish take-profit and stop-loss levels."""
    take_profit = entry_price + (grid_size * 2)  # Example: 2 grids above entry
    stop_loss = entry_price - (grid_size * 2)    # Example: 2 grids below entry
    return take_profit, stop_loss

def calculate_grids_and_leverage(predicted_low, stop_loss, grid_size):
    """Calculate the number of grids and leverage."""
    number_of_grids = (stop_loss - predicted_low) // grid_size
    leverage = 1 / ((predicted_low - stop_loss) / predicted_low)
    return number_of_grids, leverage

def define_grid_trading_strategy(predicted_low, predicted_high, entry_price, grid_count):
    """Define the grid trading strategy."""
    upper_limit, lower_limit, grid_size = set_grid_limits(predicted_low, predicted_high, grid_count)
    take_profit, stop_loss = calculate_take_profit_and_stop_loss(entry_price, grid_size)
    number_of_grids, leverage = calculate_grids_and_leverage(predicted_low, stop_loss, grid_size)

    print(f"Grid Trading Strategy:")
    print(f"  Upper Limit: {upper_limit:.2f} USD")
    print(f"  Lower Limit: {lower_limit:.2f} USD")
    print(f"  Take Profit Level: {take_profit:.2f} USD")
    print(f"  Stop Loss Level: {stop_loss:.2f} USD")
    print(f"  Number of Grids: {number_of_grids}")
    print(f"  Suggested Leverage: {leverage:.2f}x")

def calculate_grid_strategy(predicted_low, predicted_high):
    grid_count = 10  # Example grid count, replace with your logic
    leverage = 1 / ((predicted_high - predicted_low) / predicted_high)  # Example leverage calculation
    return grid_count, leverage

# Example usage (replace with actual values from analysis)
predicted_low = 91683.90
predicted_high = 99116.18
entry_price = 95000.00  # Example entry price
grid_count = 10

define_grid_trading_strategy(predicted_low, predicted_high, entry_price, grid_count)