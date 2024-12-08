import pandas as pd
import numpy as np
from scipy import stats

def calculate_grid_strategy(predicted_low, predicted_high):
    """
    Calculate aggressive grid trading strategy parameters with higher risk/reward
    
    Args:
        predicted_low (float): Predicted lower price bound
        predicted_high (float): Predicted upper price bound
        
    Returns:
        tuple: (grid_count, leverage)
    """
    # Calculate price range and average price
    price_range = predicted_high - predicted_low
    avg_price = (predicted_high + predicted_low) / 2
    
    # More aggressive volatility-based grid count
    volatility_ratio = price_range / avg_price
    base_grid_count = int(150 * volatility_ratio)  # Increased multiplier for more grids
    
    # Ensure grid count is within bounds (10-200) but favor higher counts
    grid_count = max(50, min(200, base_grid_count))
    
    # Calculate aggressive leverage based on price range and market momentum
    price_ratio = predicted_high / predicted_low
    momentum_factor = 1.5  # Aggressive momentum multiplier
    raw_leverage = (price_ratio - 1) * momentum_factor
    
    # Apply aggressive risk adjustments
    risk_factor = 0.9  # More aggressive risk factor (was 0.7)
    adjusted_leverage = raw_leverage * risk_factor
    
    # Ensure leverage is within bounds but favor higher values
    safe_leverage = max(5, min(15, adjusted_leverage))  # Minimum leverage increased to 5x
    
    return grid_count, safe_leverage

def validate_grid_parameters(grid_count, leverage):
    """Validate grid trading parameters"""
    if not (10 <= grid_count <= 200):
        return False
    if not (2 <= leverage <= 15):
        return False
    return True