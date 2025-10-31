"""
Quick test script to verify ML training works with mock data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_mock_price_data(days=365, base_price=95000):
    """Generate mock Bitcoin price data"""
    timestamps = []
    data = []
    
    current_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 24):  # Hourly data
        timestamps.append(current_time)
        
        # Add some volatility
        price_change = np.random.randn() * 0.02
        price = base_price * (1 + price_change)
        
        high = price * 1.01
        low = price * 0.99
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
        
        base_price = price
        current_time += timedelta(hours=1)
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def test_deep_learning():
    """Test deep learning model"""
    print("\n" + "="*60)
    print("🧪 Testing Deep Learning Model")
    print("="*60)
    
    from deep_learning.model import create_model
    from deep_learning.trainer import DeepLearningTrainer
    
    # Generate mock data
    print("\nGenerating mock data...")
    df = generate_mock_price_data(days=30)
    
    # Prepare sequences
    feature_data = df[['close']].values
    sequence_length = 60
    
    X, y = [], []
    for i in range(sequence_length, len(feature_data)):
        X.append(feature_data[i-sequence_length:i])
        y.append(feature_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating LSTM model...")
    model = create_model(
        model_type='lstm',
        input_size=1,
        hidden_layers=[32, 16],
        dropout=0.2
    )
    
    print(f"Model created: {model.__class__.__name__}")
    
    # Create trainer
    config = {
        'learning_rate': 0.001,
        'early_stopping_patience': 5
    }
    
    trainer = DeepLearningTrainer(model, config, device='cpu')
    
    # Prepare data loaders
    train_loader, val_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val, batch_size=16
    )
    
    # Train for just 3 epochs as a test
    print("\nTraining model (3 epochs)...")
    trainer.train(
        train_loader, val_loader,
        epochs=3,
        checkpoint_dir='test_checkpoints',
        save_interval=1
    )
    
    print("\n✅ Deep Learning test completed successfully!")
    
    # Cleanup
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')

def test_reinforcement_learning():
    """Test RL environment"""
    print("\n" + "="*60)
    print("🧪 Testing Reinforcement Learning Environment")
    print("="*60)
    
    from reinforcement_learning.trading_env import BitcoinTradingEnv
    
    # Generate mock data
    print("\nGenerating mock data...")
    df = generate_mock_price_data(days=30)
    
    # Create environment
    print("\nCreating trading environment...")
    env = BitcoinTradingEnv(
        data=df,
        initial_balance=10000,
        commission=0.001
    )
    
    # Test environment
    print("\nTesting environment for 100 steps...")
    obs, info = env.reset()
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nSteps completed: {step+1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    
    # Get metrics
    metrics = env.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    
    print("\n✅ Reinforcement Learning test completed successfully!")

if __name__ == "__main__":
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                                 ║")
    print("║   🧪 ML Version Quick Test Suite 🧪                            ║")
    print("║                                                                 ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    try:
        test_deep_learning()
        test_reinforcement_learning()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
