@echo off
REM One-click setup script for ML version (Windows)

echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                                 ║
echo ║   🚀 Bitcoin Trading ML - One-Click Setup 🚀                   ║
echo ║                                                                 ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating directory structure...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\news 2>nul
mkdir models 2>nul
mkdir checkpoints\deep_learning 2>nul
mkdir checkpoints\reinforcement_learning 2>nul
mkdir logs\deep_learning 2>nul
mkdir logs\reinforcement_learning 2>nul

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                                 ║
echo ║   ✅ Setup Complete! ✅                                        ║
echo ║                                                                 ║
echo ║   To start training:                                           ║
echo ║   1. Activate virtual environment: venv\Scripts\activate       ║
echo ║   2. Run training script: python train.py                      ║
echo ║                                                                 ║
echo ╚═══════════════════════════════════════════════════════════════╝

pause
