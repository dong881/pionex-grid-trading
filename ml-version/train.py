#!/usr/bin/env python3
"""
Main training script for ML version.
Provides interactive menu to choose training mode.
"""
import os
import sys
import yaml
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                                 ║
║   🤖 Bitcoin Trading ML Training System 🤖                     ║
║                                                                 ║
║   Powered by Deep Learning & Reinforcement Learning            ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print training mode menu"""
    menu = """
╔═══════════════════════════════════════════════════════════════╗
║                    TRAINING MODE SELECTION                      ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                 ║
║  [1] 📊 Collect Historical Data                                ║
║      Download historical Bitcoin price data                    ║
║                                                                 ║
║  [2] 📰 Collect News Data                                      ║
║      Collect and analyze Bitcoin news sentiment                ║
║                                                                 ║
║  [3] 🧠 Deep Learning Training                                 ║
║      Train LSTM/GRU model with price + news + indicators       ║
║                                                                 ║
║  [4] 🎮 Reinforcement Learning Training                        ║
║      Train RL agent to learn trading strategies                ║
║                                                                 ║
║  [5] 📈 Evaluate Models                                        ║
║      Test and evaluate trained models                          ║
║                                                                 ║
║  [6] 🔄 Full Pipeline                                          ║
║      Run complete data collection → training pipeline          ║
║                                                                 ║
║  [0] ❌ Exit                                                   ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(menu)

def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    else:
        print(f"⚠️  Warning: Config file not found at {config_path}")
        return {}

def collect_historical_data(config):
    """Collect historical price data"""
    print("\n" + "="*60)
    print("📊 COLLECTING HISTORICAL DATA")
    print("="*60)
    
    from data_collection.historical_data import HistoricalDataCollector
    
    data_config = config.get('data', {})
    lookback_days = data_config.get('lookback_days', 365)
    
    collector = HistoricalDataCollector()
    collector.collect_all_intervals(days=lookback_days)
    
    print("\n✅ Historical data collection complete!")

def collect_news_data(config):
    """Collect news data and analyze sentiment"""
    print("\n" + "="*60)
    print("📰 COLLECTING NEWS DATA")
    print("="*60)
    
    from data_collection.news_collector import NewsCollector
    from data_collection.sentiment_analyzer import SentimentAnalyzer
    
    news_config = config.get('news', {})
    keywords = news_config.get('keywords', ['bitcoin', 'btc'])
    lookback_days = news_config.get('lookback_days', 30)
    
    # Collect news
    collector = NewsCollector()
    articles = collector.collect_all(keywords=keywords, days=lookback_days)
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    analyzed_articles = analyzer.analyze_articles(articles)
    
    # Save
    output_path = os.path.join('data', 'news', f'analyzed_news_{datetime.now().strftime("%Y%m%d")}.json')
    analyzer.save_analyzed_articles(analyzed_articles, output_path)
    
    print("\n✅ News data collection and analysis complete!")

def train_deep_learning(config):
    """Train deep learning model"""
    print("\n" + "="*60)
    print("🧠 DEEP LEARNING TRAINING")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    from deep_learning.model import create_model
    from deep_learning.trainer import DeepLearningTrainer
    
    # Load data
    data_file = os.path.join('data', 'raw', 'btc_1h_*.csv')
    import glob
    files = glob.glob(data_file)
    
    if not files:
        print("❌ No historical data found. Please collect data first (option 1).")
        return
    
    print(f"Loading data from {files[0]}")
    df = pd.read_csv(files[0], index_col=0, parse_dates=True)
    
    # Prepare features (simplified for demo)
    feature_data = df[['close']].values
    
    # Create sequences
    sequence_length = config.get('deep_learning', {}).get('sequence_length', 60)
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
    
    print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
    
    # Create model
    dl_config = config.get('deep_learning', {})
    model = create_model(
        model_type=dl_config.get('model_type', 'lstm'),
        input_size=X.shape[2],
        hidden_layers=dl_config.get('hidden_layers', [128, 64, 32]),
        dropout=dl_config.get('dropout', 0.2)
    )
    
    # Train
    trainer = DeepLearningTrainer(model, dl_config, device='cpu')
    
    train_loader, val_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val,
        batch_size=dl_config.get('batch_size', 32)
    )
    
    trainer.train(
        train_loader, val_loader,
        epochs=dl_config.get('epochs', 100),
        checkpoint_dir='checkpoints/deep_learning',
        save_interval=10
    )
    
    print("\n✅ Deep learning training complete!")

def train_reinforcement_learning(config):
    """Train reinforcement learning model"""
    print("\n" + "="*60)
    print("🎮 REINFORCEMENT LEARNING TRAINING")
    print("="*60)
    
    import pandas as pd
    from reinforcement_learning.trainer import RLTrainer
    
    # Load data
    data_file = os.path.join('data', 'raw', 'btc_1h_*.csv')
    import glob
    files = glob.glob(data_file)
    
    if not files:
        print("❌ No historical data found. Please collect data first (option 1).")
        return
    
    print(f"Loading data from {files[0]}")
    df = pd.read_csv(files[0], index_col=0, parse_dates=True)
    
    # Create trainer
    trainer = RLTrainer(df, config)
    
    # Create model
    rl_config = config.get('reinforcement_learning', {})
    algorithm = rl_config.get('algorithm', 'ppo')
    trainer.create_model(algorithm=algorithm)
    
    # Train
    training_config = rl_config.get('training', {})
    trainer.train(
        total_timesteps=training_config.get('total_timesteps', 1000000),
        checkpoint_dir='checkpoints/reinforcement_learning',
        log_dir='logs/reinforcement_learning'
    )
    
    # Evaluate
    print("\nEvaluating trained model...")
    trainer.evaluate(num_episodes=5)
    
    print("\n✅ Reinforcement learning training complete!")

def run_full_pipeline(config):
    """Run complete pipeline"""
    print("\n" + "="*60)
    print("🔄 RUNNING FULL PIPELINE")
    print("="*60)
    
    print("\nStep 1/4: Collecting historical data...")
    collect_historical_data(config)
    
    print("\nStep 2/4: Collecting news data...")
    collect_news_data(config)
    
    print("\nStep 3/4: Training deep learning model...")
    train_deep_learning(config)
    
    print("\nStep 4/4: Training reinforcement learning model...")
    train_reinforcement_learning(config)
    
    print("\n✅ Full pipeline complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Bitcoin Trading ML Training System')
    parser.add_argument('--mode', type=int, help='Training mode (1-6)', default=None)
    args = parser.parse_args()
    
    print_banner()
    
    # Load configuration
    config = load_config()
    
    # Interactive mode if no mode specified
    if args.mode is None:
        while True:
            print_menu()
            choice = input("Select an option (0-6): ").strip()
            
            try:
                choice = int(choice)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
                continue
            
            if choice == 0:
                print("\n👋 Goodbye!")
                break
            elif choice == 1:
                collect_historical_data(config)
            elif choice == 2:
                collect_news_data(config)
            elif choice == 3:
                train_deep_learning(config)
            elif choice == 4:
                train_reinforcement_learning(config)
            elif choice == 5:
                print("📈 Evaluation feature coming soon!")
            elif choice == 6:
                run_full_pipeline(config)
            else:
                print("❌ Invalid option. Please select 0-6.")
            
            input("\nPress Enter to continue...")
    else:
        # Non-interactive mode
        if args.mode == 1:
            collect_historical_data(config)
        elif args.mode == 2:
            collect_news_data(config)
        elif args.mode == 3:
            train_deep_learning(config)
        elif args.mode == 4:
            train_reinforcement_learning(config)
        elif args.mode == 6:
            run_full_pipeline(config)

if __name__ == "__main__":
    main()
