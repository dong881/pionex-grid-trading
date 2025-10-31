"""
Reinforcement Learning trainer for Bitcoin trading.
Uses Stable-Baselines3 for RL algorithms.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
import json

try:
    from stable_baselines3 import PPO, A2C, DQN, SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available")

from .trading_env import BitcoinTradingEnv

class TradingCallback(BaseCallback):
    """Custom callback for logging training progress"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super(TradingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called after each step"""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Log metrics
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            print(f"Rollout {self.num_timesteps}: Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.0f}")

class RLTrainer:
    """Trainer for reinforcement learning models"""
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        """
        Initialize RL trainer
        
        Args:
            data: Historical price data
            config: Configuration dictionary
        """
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RL training")
        
        self.data = data
        self.config = config
        self.env = None
        self.model = None
    
    def create_environment(self) -> BitcoinTradingEnv:
        """Create trading environment"""
        env_config = self.config.get('environment', {})
        
        env = BitcoinTradingEnv(
            data=self.data,
            initial_balance=env_config.get('initial_balance', 10000),
            commission=env_config.get('commission', 0.001)
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        return env
    
    def create_model(self, algorithm: str = 'ppo'):
        """
        Create RL model
        
        Args:
            algorithm: RL algorithm ('ppo', 'a2c', 'dqn', 'sac')
        """
        self.env = DummyVecEnv([self.create_environment])
        
        training_config = self.config.get('training', {})
        
        algorithm = algorithm.lower()
        
        if algorithm == 'ppo':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=training_config.get('learning_rate', 0.0003),
                n_steps=training_config.get('n_steps', 2048),
                batch_size=training_config.get('batch_size', 64),
                gamma=training_config.get('gamma', 0.99),
                verbose=1
            )
        elif algorithm == 'a2c':
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=training_config.get('learning_rate', 0.0003),
                gamma=training_config.get('gamma', 0.99),
                verbose=1
            )
        elif algorithm == 'dqn':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=training_config.get('learning_rate', 0.0001),
                batch_size=training_config.get('batch_size', 64),
                gamma=training_config.get('gamma', 0.99),
                verbose=1
            )
        elif algorithm == 'sac':
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=training_config.get('learning_rate', 0.0003),
                batch_size=training_config.get('batch_size', 64),
                gamma=training_config.get('gamma', 0.99),
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"\n{algorithm.upper()} model created")
    
    def train(self, total_timesteps: int = 1000000, 
             checkpoint_dir: str = 'checkpoints',
             log_dir: str = 'logs'):
        """
        Train the RL model
        
        Args:
            total_timesteps: Total training timesteps
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix='rl_model'
        )
        
        trading_callback = TradingCallback(log_dir=log_dir)
        
        # Train
        print(f"\nStarting RL training for {total_timesteps} timesteps...")
        print(f"{'='*60}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, trading_callback]
        )
        
        print(f"\n{'='*60}")
        print("Training complete!")
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, 'final_model.zip')
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        print(f"\nEvaluating model for {num_episodes} episodes...")
        
        all_metrics = []
        
        for episode in range(num_episodes):
            env = self.create_environment()
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            metrics = env.get_metrics()
            all_metrics.append(metrics)
            
            print(f"Episode {episode+1}: Return: {metrics['total_return']:.2f}%, "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                  f"Max DD: {metrics['max_drawdown']:.2f}%")
        
        # Calculate average metrics
        avg_metrics = {
            'avg_return': np.mean([m['total_return'] for m in all_metrics]),
            'avg_sharpe': np.mean([m['sharpe_ratio'] for m in all_metrics]),
            'avg_max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'avg_trades': np.mean([m['total_trades'] for m in all_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics])
        }
        
        print(f"\n{'='*60}")
        print("Evaluation Results:")
        print(f"Average Return: {avg_metrics['avg_return']:.2f}%")
        print(f"Average Sharpe Ratio: {avg_metrics['avg_sharpe']:.2f}")
        print(f"Average Max Drawdown: {avg_metrics['avg_max_drawdown']:.2f}%")
        print(f"Average Trades: {avg_metrics['avg_trades']:.0f}")
        print(f"Average Win Rate: {avg_metrics['avg_win_rate']:.2f}%")
        
        return avg_metrics
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        algorithm = self.config.get('reinforcement_learning', {}).get('algorithm', 'ppo')
        
        self.env = DummyVecEnv([self.create_environment])
        
        if algorithm == 'ppo':
            self.model = PPO.load(filepath, env=self.env)
        elif algorithm == 'a2c':
            self.model = A2C.load(filepath, env=self.env)
        elif algorithm == 'dqn':
            self.model = DQN.load(filepath, env=self.env)
        elif algorithm == 'sac':
            self.model = SAC.load(filepath, env=self.env)
        
        print(f"Model loaded from {filepath}")
