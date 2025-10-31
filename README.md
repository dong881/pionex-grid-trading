# Pionex Grid Trading Strategy & AI Trading System

This project provides **two complete systems** for Bitcoin trading analysis and strategy development:

1. **🌐 Web Version**: Real-time grid trading dashboard with GitHub Pages deployment
2. **🤖 ML Version**: AI-powered trading with deep learning and reinforcement learning

---

## 🌐 Web Version - Grid Trading Dashboard

A web-based dashboard that automatically analyzes Bitcoin grid trading strategies and displays performance metrics. **Completely free** using public APIs and GitHub Pages hosting.

### Features
- 📊 Real-time grid trading analysis
- 📉 Drawdown calculations and visualization
- 🔄 Auto-updates every 4 hours via GitHub Actions
- 💰 100% free (no API keys required)
- 🌍 Accessible from anywhere

### Quick Start

**View Live Dashboard:**
1. Enable GitHub Pages in repository settings
2. Set Pages source to "GitHub Actions"
3. Visit: `https://[your-username].github.io/pionex-grid-trading/`

**Local Testing:**
```bash
cd web-version
python -m http.server 8000
# Visit http://localhost:8000
```

### How It Works
- GitHub Actions runs every 4 hours
- Fetches Bitcoin price data from Pionex API (free)
- Calculates optimal grid parameters and drawdown
- Generates static JSON files
- Deploys to GitHub Pages automatically

[📖 Full Web Version Documentation →](web-version/README.md)

---

## 🤖 ML Version - AI Trading System

A comprehensive machine learning system that trains AI models to predict Bitcoin prices and learn optimal trading strategies using historical data and news sentiment.

### Features

#### 🧠 Deep Learning
- LSTM/GRU/Transformer models
- Historical price + technical indicators + news sentiment
- Automatic checkpointing and model backup
- TensorBoard visualization

#### 🎮 Reinforcement Learning
- PPO, A2C, DQN, SAC algorithms
- Custom Bitcoin trading environment
- Learn strategies from historical market data
- Performance tracking (Sharpe ratio, drawdown, win rate)

#### 📊 Data Collection
- Historical Bitcoin data (Binance - free)
- News sentiment analysis (GNews - free)
- Multiple timeframes (1h, 4h, 1d)

### Quick Start (One-Click Setup)

**Linux/Mac:**
```bash
cd ml-version
./setup.sh
python train.py
```

**Windows:**
```batch
cd ml-version
setup.bat
python train.py
```

### Training Options

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
╚═══════════════════════════════════════════════════════════════╝
```

[📖 Full ML Version Documentation →](ml-version/README.md)

---

## 📁 Project Structure

```
pionex-grid-trading/
├── web-version/              # Web dashboard (GitHub Pages)
│   ├── index.html           # Dashboard interface
│   ├── static/              # CSS, JS, and data files
│   ├── api/                 # Data generation scripts
│   └── README.md
│
├── ml-version/              # ML training system
│   ├── train.py            # Main training script
│   ├── setup.sh/bat        # One-click setup
│   ├── config/             # Configuration files
│   ├── src/                # Source code
│   │   ├── data_collection/
│   │   ├── deep_learning/
│   │   ├── reinforcement_learning/
│   │   └── visualization/
│   ├── data/               # Training data
│   ├── models/             # Saved models
│   ├── checkpoints/        # Training checkpoints
│   └── README.md
│
├── src/                     # Original grid trading scripts
│   ├── fetch_data.py
│   ├── analysis.py
│   ├── strategy.py
│   └── main.py
│
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions workflow
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Which Version Should I Use?

### Use **Web Version** if you want to:
- ✅ Monitor grid trading strategies online
- ✅ Access from any device with a browser
- ✅ No local setup required
- ✅ Free hosting on GitHub Pages
- ✅ Automatic updates every 4 hours

### Use **ML Version** if you want to:
- ✅ Train custom AI models
- ✅ Predict Bitcoin price movements
- ✅ Develop automated trading strategies
- ✅ Analyze news sentiment impact
- ✅ Use reinforcement learning
- ✅ GPU-accelerated training

### Use **Both** for:
- ✅ Complete trading system
- ✅ Online monitoring + AI predictions
- ✅ Strategy validation and backtesting
- ✅ Maximum flexibility

---

## 🛠️ Original Grid Trading Script

The original implementation is still available in the `src/` folder:

```bash
# Install dependencies
pip install -r requirements.txt

# Run grid trading analysis
python src/main.py
```

This provides a simple command-line interface for grid trading analysis.

---

## 📊 Free APIs Used

All APIs used are **completely free** with no registration required:

- **Pionex API**: Grid trading price data
- **Binance API**: Historical OHLCV data
- **GNews**: Bitcoin news articles
- **CryptoPanic**: Cryptocurrency news

Optional premium APIs can be configured using GitHub Secrets.

---

## 🔧 System Requirements

### Web Version
- None (runs in browser)
- GitHub account for Pages hosting

### ML Version
- Python 3.8+
- 8GB RAM (16GB recommended)
- GPU optional (recommended for faster training)
- 10GB disk space

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional ML models and algorithms
- More data sources
- Enhanced visualization
- Trading strategy variations
- Performance optimizations

Feel free to submit issues or pull requests.

---

## 📝 License

MIT License - Free to use and modify

---

## 🙏 Acknowledgments

- Pionex for public API access
- Binance for historical data
- Stable-Baselines3 for RL implementations
- PyTorch and TensorFlow communities

---

## 📞 Support

For questions or issues:
1. Check the documentation in each version's README
2. Review existing GitHub issues
3. Open a new issue with detailed information

---

**Ready to get started?**
- [Web Version Setup →](web-version/README.md)
- [ML Version Setup →](ml-version/README.md)