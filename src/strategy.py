import pandas as pd
import numpy as np
from scipy import stats

def calculate_grid_strategy(predicted_low, predicted_high):
    """
    Calculate optimal grid trading strategy parameters based on price range analysis
    
    Args:
        predicted_low (float): Predicted lower price bound
        predicted_high (float): Predicted upper price bound
        
    Returns:
        tuple: (grid_count, leverage)
    """
    # Calculate price range and average price
    price_range = predicted_high - predicted_low
    avg_price = (predicted_high + predicted_low) / 2
    
    # Calculate volatility-based grid count
    volatility_ratio = price_range / avg_price
    base_grid_count = int(100 * volatility_ratio)  # Scale grids with volatility
    
    # Ensure grid count is within bounds (10-200)
    grid_count = max(10, min(200, base_grid_count))
    
    # Calculate optimal leverage based on price range and risk metrics
    price_ratio = predicted_high / predicted_low
    raw_leverage = price_ratio - 1  # Base leverage on price movement potential
    
    # Apply risk adjustments
    risk_factor = 0.7  # Conservative risk factor
    adjusted_leverage = raw_leverage * risk_factor
    
    # Ensure leverage is within safe bounds (2-15x)
    safe_leverage = max(2, min(15, adjusted_leverage))
    
    return grid_count, safe_leverage

def validate_grid_parameters(grid_count, leverage):
    """Validate grid trading parameters"""
    if not (10 <= grid_count <= 200):
        return False
    if not (2 <= leverage <= 15):
        return False
    return True