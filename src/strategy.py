import pandas as pd
import numpy as np
from scipy import stats

def calculate_grid_strategy(current_price, predicted_high):
    """
    Calculate optimal grid strategy based on current market position
    
    Args:
        current_price (float): Current market price
        predicted_high (float): Predicted upper price bound
        
    Returns:
        tuple: (grid_count, leverage, upper_limit, lower_limit)
    """
    # Calculate dynamic price ranges based on current price
    price_volatility = (predicted_high - current_price) / current_price
    
    # Calculate aggressive bounds
    upper_limit = current_price * (1 + price_volatility * 1.2)  # 20% more volatile upside
    lower_limit = current_price * (1 - price_volatility * 0.8)  # 20% less volatile downside
    
    # Calculate grid parameters
    price_range = upper_limit - lower_limit
    volatility_ratio = price_range / current_price
    
    # More aggressive grid count calculation
    base_grid_count = int(price_range / 100)  # One grid per $100 range
    grid_count = max(10, min(200, base_grid_count))
    
    # Calculate aggressive leverage
    price_ratio = upper_limit / lower_limit
    momentum_factor = 1.8  # More aggressive momentum multiplier
    raw_leverage = (price_ratio - 1) * momentum_factor
    
    # Risk adjustments
    risk_factor = 0.95  # Very aggressive risk factor
    adjusted_leverage = raw_leverage * risk_factor
    leverage = max(5, min(15, adjusted_leverage))
    
    return grid_count, leverage, upper_limit, lower_limit

def validate_grid_parameters(grid_count, leverage, upper_limit, lower_limit):
    """Validate grid trading parameters"""
    if not (10 <= grid_count <= 200):
        return False
    if not (5 <= leverage <= 15):
        return False
    if upper_limit <= lower_limit:
        return False
    return True