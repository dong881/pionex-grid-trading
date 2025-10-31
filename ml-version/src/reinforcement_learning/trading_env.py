"""
Reinforcement Learning trading environment for Bitcoin.
Uses Gymnasium API for RL training.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class BitcoinTradingEnv(gym.Env):
    """Custom Trading Environment for Bitcoin"""
    
    metadata = {'render_modes': ['human']}
    
    # Numerical stability constant to prevent division by zero
    EPSILON = 1e-8
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000,
                 commission: float = 0.001, window_size: int = 60):
        """
        Initialize trading environment
        
        Args:
            data: Historical price data (DataFrame with OHLCV)
            initial_balance: Initial account balance in USDT
            commission: Trading commission rate
            window_size: Number of past observations to include in state
        """
        super(BitcoinTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        # Trading state
        self.balance = initial_balance
        self.position = 0.0  # Current BTC position
        self.entry_price = 0.0
        self.current_step = window_size
        self.max_steps = len(data) - 1
        
        # Performance tracking
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.portfolio_values = []
        
        # Action space: [hold, buy, sell]
        # 0: hold, 1: buy 25%, 2: buy 50%, 3: buy 100%,
        # 4: sell 25%, 5: sell 50%, 6: sell 100%
        self.action_space = spaces.Discrete(7)
        
        # Observation space: [price features, technical indicators, account info]
        # Price window + position + balance + profit
        obs_size = window_size * 5 + 3  # OHLCV * window + 3 account features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get price window
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalize price data
        price_features = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in window_data.columns:
                values = window_data[col].values
                # Normalize by dividing by the first value
                normalized = values / (values[0] + self.EPSILON)
                price_features.extend(normalized)
        
        # Account features (normalized)
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        account_features = [
            self.position / 10.0,  # Normalized position
            self.balance / self.initial_balance,  # Normalized balance
            portfolio_value / self.initial_balance  # Normalized portfolio value
        ]
        
        observation = np.array(price_features + account_features, dtype=np.float32)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0-6)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = 0
        trade_executed = False
        
        if action == 0:  # Hold
            pass
        
        elif action in [1, 2, 3]:  # Buy
            buy_percentages = {1: 0.25, 2: 0.50, 3: 1.0}
            buy_amount = self.balance * buy_percentages[action]
            
            if buy_amount > 0:
                commission_cost = buy_amount * self.commission
                btc_amount = (buy_amount - commission_cost) / current_price
                
                self.position += btc_amount
                self.balance -= buy_amount
                self.entry_price = current_price
                self.total_trades += 1
                trade_executed = True
        
        elif action in [4, 5, 6]:  # Sell
            sell_percentages = {4: 0.25, 5: 0.50, 6: 1.0}
            sell_amount = self.position * sell_percentages[action]
            
            if sell_amount > 0:
                sell_value = sell_amount * current_price
                commission_cost = sell_value * self.commission
                
                profit = (current_price - self.entry_price) * sell_amount
                self.total_profit += profit
                
                if profit > 0:
                    self.winning_trades += 1
                    reward += profit / self.initial_balance  # Normalized profit
                
                self.balance += sell_value - commission_cost
                self.position -= sell_amount
                self.total_trades += 1
                trade_executed = True
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Reward calculation
        returns = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        reward += returns * 100  # Scale up the reward
        
        # Penalize holding penalty
        if action == 0 and self.position > 0:
            reward -= 0.001  # Small penalty for holding
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Info
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'trade_executed': trade_executed
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        portfolio_value = self.balance + (self.position * self.data.iloc[self.current_step]['close'])
        print(f"Step: {self.current_step}, "
              f"Balance: ${self.balance:.2f}, "
              f"Position: {self.position:.4f} BTC, "
              f"Portfolio: ${portfolio_value:.2f}")
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + self.EPSILON) * np.sqrt(252)
        
        # Calculate max drawdown
        cummax = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - cummax) / cummax
        max_drawdown = np.min(drawdown) * 100
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate
        }
