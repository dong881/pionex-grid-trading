"""
News data collection module for Bitcoin sentiment analysis.
Collects news articles from various sources and analyzes their sentiment.
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from tqdm import tqdm

class NewsCollector:
    """Collects news articles related to Bitcoin"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/news"):
        """
        Initialize NewsCollector
        
        Args:
            api_key: API key for news services (optional)
            output_dir: Directory to save collected news
        """
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_gnews(self, keywords: List[str], days: int = 30) -> List[Dict]:
        """
        Collect news from GNews (free, no API key required)
        
        Args:
            keywords: List of keywords to search
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        articles = []
        
        try:
            from gnews import GNews
            
            google_news = GNews(
                language='en',
                country='US',
                period=f'{days}d',
                max_results=100
            )
            
            for keyword in keywords:
                print(f"Collecting news for: {keyword}")
                news = google_news.get_news(keyword)
                
                for item in news:
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'published_date': item.get('published date', ''),
                        'url': item.get('url', ''),
                        'source': 'gnews',
                        'keyword': keyword
                    })
        except ImportError:
            print("Warning: gnews package not installed. Install with: pip install gnews")
        except Exception as e:
            print(f"Error collecting from GNews: {e}")
        
        return articles
    
    def collect_from_free_api(self, keywords: List[str], days: int = 30) -> List[Dict]:
        """
        Collect news from free cryptocurrency news API
        
        Args:
            keywords: List of keywords to search
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        articles = []
        
        # CryptoPanic API (free tier available)
        base_url = "https://cryptopanic.com/api/v1/posts/"
        
        try:
            params = {
                'auth_token': 'public',  # Public access
                'currencies': 'BTC',
                'filter': 'news',
                'public': 'true'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('results', []):
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('title', ''),  # Use title as description
                        'published_date': item.get('created_at', ''),
                        'url': item.get('url', ''),
                        'source': 'cryptopanic',
                        'keyword': 'bitcoin'
                    })
        except Exception as e:
            print(f"Error collecting from free API: {e}")
        
        return articles
    
    def collect_all(self, keywords: List[str] = None, days: int = 30) -> List[Dict]:
        """
        Collect news from all available sources
        
        Args:
            keywords: List of keywords to search
            days: Number of days to look back
            
        Returns:
            List of all collected news articles
        """
        if keywords is None:
            keywords = ['bitcoin', 'btc', 'cryptocurrency']
        
        print(f"\nCollecting news for the last {days} days...")
        all_articles = []
        
        # Collect from GNews
        print("\n[1/2] Collecting from GNews...")
        gnews_articles = self.collect_gnews(keywords, days)
        all_articles.extend(gnews_articles)
        print(f"Collected {len(gnews_articles)} articles from GNews")
        
        # Collect from free API
        print("\n[2/2] Collecting from free APIs...")
        api_articles = self.collect_from_free_api(keywords, days)
        all_articles.extend(api_articles)
        print(f"Collected {len(api_articles)} articles from free APIs")
        
        # Remove duplicates based on title
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        print(f"\nTotal unique articles: {len(unique_articles)}")
        
        return unique_articles
    
    def save_articles(self, articles: List[Dict], filename: str = None):
        """
        Save collected articles to JSON file
        
        Args:
            articles: List of articles to save
            filename: Output filename (default: news_YYYYMMDD.json)
        """
        if filename is None:
            filename = f"news_{datetime.now().strftime('%Y%m%d')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'collected_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'articles': articles
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(articles)} articles to {filepath}")
        return filepath

def main():
    """Main function for testing"""
    collector = NewsCollector()
    articles = collector.collect_all(days=7)
    collector.save_articles(articles)

if __name__ == "__main__":
    main()
