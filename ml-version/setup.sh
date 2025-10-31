#!/bin/bash

# One-click setup script for ML version

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                                 ║"
echo "║   🚀 Bitcoin Trading ML - One-Click Setup 🚀                   ║"
echo "║                                                                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/{raw,processed,news}
mkdir -p models
mkdir -p checkpoints/{deep_learning,reinforcement_learning}
mkdir -p logs/{deep_learning,reinforcement_learning}

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                                 ║"
echo "║   ✅ Setup Complete! ✅                                        ║"
echo "║                                                                 ║"
echo "║   To start training:                                           ║"
echo "║   1. Activate virtual environment: source venv/bin/activate    ║"
echo "║   2. Run training script: python train.py                      ║"
echo "║                                                                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
