"""
Historical price data collection module.
Collects Bitcoin historical data from free APIs.
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict
import requests
import pandas as pd

class HistoricalDataCollector:
    """Collects historical Bitcoin price data"""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize data collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_binance_data(self, symbol: str = "BTCUSDT", interval: str = "1h", 
                          start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch historical data from Binance (free, no API key required)
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with historical data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        base_url = "https://api.binance.com/api/v3/klines"
        
        all_data = []
        current_start = start_date
        
        print(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}...")
        
        while current_start < end_date:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': 1000  # Max limit per request
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        break
                    
                    all_data.extend(data)
                    
                    # Update start time to the last timestamp + 1
                    last_timestamp = data[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
                    
                    print(f"Fetched {len(data)} records, total: {len(all_data)}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                else:
                    print(f"Error: HTTP {response.status_code}")
                    break
            
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Keep only relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
        
        return pd.DataFrame()
    
    def fetch_coinbase_data(self, symbol: str = "BTC-USD", granularity: int = 3600) -> pd.DataFrame:
        """
        Fetch historical data from Coinbase Pro (free, no API key required)
        
        Args:
            symbol: Trading pair symbol
            granularity: Time interval in seconds (60, 300, 900, 3600, 21600, 86400)
            
        Returns:
            DataFrame with historical data
        """
        base_url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        
        all_data = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        
        print(f"Fetching {symbol} data from Coinbase...")
        
        # Coinbase allows max 300 candles per request
        current_end = end_time
        
        while current_end > start_time:
            params = {
                'granularity': granularity,
                'end': current_end.isoformat(),
                'start': (current_end - timedelta(days=30)).isoformat()
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        break
                    
                    all_data.extend(data)
                    
                    print(f"Fetched {len(data)} records, total: {len(all_data)}")
                    
                    # Update end time
                    current_end = current_end - timedelta(days=30)
                    
                    # Rate limiting
                    time.sleep(1)
                else:
                    print(f"Error: HTTP {response.status_code}")
                    break
            
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
        
        return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """
        Save collected data to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if filename is None:
            filename = f"btc_historical_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath)
        
        print(f"\nSaved {len(df)} records to {filepath}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return filepath
    
    def collect_all_intervals(self, days: int = 365):
        """
        Collect data for multiple intervals
        
        Args:
            days: Number of days to collect
        """
        intervals = {
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        for name, interval in intervals.items():
            print(f"\n{'='*60}")
            print(f"Collecting {name} data...")
            print(f"{'='*60}")
            
            df = self.fetch_binance_data(
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                filename = f"btc_{name}_{datetime.now().strftime('%Y%m%d')}.csv"
                self.save_data(df, filename)
            else:
                print(f"No data collected for {name}")

def main():
    """Main function for testing"""
    collector = HistoricalDataCollector()
    collector.collect_all_intervals(days=365)

if __name__ == "__main__":
    main()
