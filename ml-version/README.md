# ML Version - Bitcoin Trading with AI

This is the machine learning version that trains AI models to predict Bitcoin prices and learn trading strategies using deep learning and reinforcement learning.

## Features

### 🧠 Deep Learning
- LSTM/GRU/Transformer models for price prediction
- Technical indicators integration
- News sentiment analysis
- Historical data training

### 🎮 Reinforcement Learning
- PPO, A2C, DQN, SAC algorithms
- Custom trading environment
- Reward optimization
- Strategy learning from historical data

### 📊 Data Collection
- Historical Bitcoin price data (Binance API - free)
- News sentiment analysis (GNews, CryptoPanic - free)
- Multiple timeframes (1h, 4h, 1d)

### 🎯 Training Features
- Automatic checkpointing
- Training progress visualization
- Model backup and versioning
- GPU/CPU support
- TensorBoard integration

## Quick Start (One-Click Setup)

### Linux/Mac:
```bash
cd ml-version
./setup.sh
```

### Windows:
```batch
cd ml-version
setup.bat
```

### Manual Setup:
```bash
cd ml-version
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Interactive Mode:
```bash
python train.py
```

You'll see a menu:
```
╔═══════════════════════════════════════════════════════════════╗
║                    TRAINING MODE SELECTION                      ║
╠═══════════════════════════════════════════════════════════════╣
║  [1] 📊 Collect Historical Data                                ║
║  [2] 📰 Collect News Data                                      ║
║  [3] 🧠 Deep Learning Training                                 ║
║  [4] 🎮 Reinforcement Learning Training                        ║
║  [5] 📈 Evaluate Models                                        ║
║  [6] 🔄 Full Pipeline                                          ║
║  [0] ❌ Exit                                                   ║
╚═══════════════════════════════════════════════════════════════╝
```

### Command Line Mode:
```bash
# Collect data
python train.py --mode 1

# Train deep learning model
python train.py --mode 3

# Train RL model
python train.py --mode 4

# Run full pipeline
python train.py --mode 6
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Deep Learning
deep_learning:
  model_type: "lstm"  # Options: lstm, gru, transformer
  sequence_length: 60
  hidden_layers: [128, 64, 32]
  epochs: 100

# Reinforcement Learning
reinforcement_learning:
  algorithm: "ppo"  # Options: ppo, a2c, dqn, sac
  total_timesteps: 1000000
```

## Project Structure

```
ml-version/
├── train.py                # Main training script
├── setup.sh/setup.bat      # Setup scripts
├── requirements.txt        # Dependencies
├── config/
│   └── config.yaml        # Configuration
├── src/
│   ├── data_collection/   # Data collection modules
│   │   ├── historical_data.py
│   │   ├── news_collector.py
│   │   └── sentiment_analyzer.py
│   ├── deep_learning/     # DL models and training
│   │   ├── model.py
│   │   └── trainer.py
│   ├── reinforcement_learning/  # RL environment and training
│   │   ├── trading_env.py
│   │   └── trainer.py
│   └── visualization/     # Training visualization
├── data/
│   ├── raw/              # Raw historical data
│   ├── processed/        # Processed features
│   └── news/             # News and sentiment data
├── models/               # Saved models
├── checkpoints/          # Training checkpoints
└── logs/                 # Training logs
```

## Training Workflows

### Option 1: Deep Learning (Price + News + Indicators)

1. **Data Collection**
   ```bash
   python train.py --mode 1  # Collect historical data
   python train.py --mode 2  # Collect news data
   ```

2. **Training**
   ```bash
   python train.py --mode 3  # Train DL model
   ```

3. **Monitoring**
   - Checkpoints saved every 10 epochs
   - Best model saved automatically
   - Training history in `checkpoints/deep_learning/training_history.json`

### Option 2: Reinforcement Learning (Learn Trading)

1. **Data Collection**
   ```bash
   python train.py --mode 1  # Collect historical data
   ```

2. **Training**
   ```bash
   python train.py --mode 4  # Train RL agent
   ```

3. **Evaluation**
   - Model automatically evaluated after training
   - Checkpoints saved every 50,000 timesteps
   - Final model saved as `final_model.zip`

## GPU Support

The system automatically detects and uses GPU if available:

```yaml
training:
  device: "cuda"  # Options: cuda (NVIDIA), mps (Apple M1/M2), cpu
```

For NVIDIA GPUs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2):
```bash
pip install torch torchvision torchaudio
# MPS backend is automatically used
```

## Model Outputs

### Deep Learning
- **Best Model**: `checkpoints/deep_learning/best_model.pt`
- **Checkpoints**: `checkpoints/deep_learning/checkpoint_epoch_*.pt`
- **History**: `checkpoints/deep_learning/training_history.json`

### Reinforcement Learning
- **Final Model**: `checkpoints/reinforcement_learning/final_model.zip`
- **Checkpoints**: `checkpoints/reinforcement_learning/rl_model_*_steps.zip`
- **Logs**: `logs/reinforcement_learning/`

## News Sentiment Analysis

The system analyzes news sentiment automatically:

- **Bullish**: Positive news (score 60-100)
- **Bearish**: Negative news (score 0-40)
- **Neutral**: Mixed sentiment (score 40-60)

News data is saved with sentiment scores in `data/news/`.

## Performance Metrics

### Deep Learning
- Training Loss
- Validation Loss
- Prediction Accuracy

### Reinforcement Learning
- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown (%)
- Win Rate (%)
- Total Trades

## Advanced Features

### TensorBoard Visualization

```bash
tensorboard --logdir=logs/
```

### Custom Strategies

Modify `src/reinforcement_learning/trading_env.py` to implement custom:
- Reward functions
- Trading actions
- Risk management

### Data Sources

Add new data sources in:
- `src/data_collection/historical_data.py`
- `src/data_collection/news_collector.py`

## Troubleshooting

**Out of Memory:**
- Reduce batch size in config
- Use CPU instead of GPU
- Reduce sequence length

**Training Too Slow:**
- Use GPU if available
- Reduce epochs or timesteps
- Use smaller model

**Poor Performance:**
- Increase training data
- Adjust hyperparameters
- Try different algorithms

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended
- Internet connection for data collection

## Free APIs Used

- **Binance API**: Historical price data (no key required)
- **GNews**: News articles (no key required)
- **CryptoPanic**: Crypto news (public tier)

## License

MIT License - See main repository LICENSE file
