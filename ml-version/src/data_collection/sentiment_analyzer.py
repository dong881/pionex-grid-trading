"""
Sentiment analysis module for news articles.
Analyzes news sentiment and assigns bullish/bearish scores.
"""
import json
import os
from typing import Dict, List
from datetime import datetime

class SentimentAnalyzer:
    """Analyzes sentiment of news articles"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sentiment_model = None
        self.use_transformers = False
        
        # Try to load transformers model (optional)
        try:
            from transformers import pipeline
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU
            )
            self.use_transformers = True
            print("Using FinBERT model for sentiment analysis")
        except ImportError:
            print("Transformers not available, using rule-based sentiment")
        except Exception as e:
            print(f"Could not load sentiment model: {e}")
            print("Using rule-based sentiment")
    
    def analyze_with_keywords(self, text: str) -> Dict:
        """
        Analyze sentiment using keyword-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment and score
        """
        text_lower = text.lower()
        
        # Bullish keywords
        bullish_keywords = [
            'bull', 'bullish', 'rise', 'rising', 'surge', 'surging', 'gain', 'gains',
            'up', 'higher', 'increase', 'increasing', 'growth', 'rally', 'rallying',
            'pump', 'pumping', 'moon', 'mooning', 'breakthrough', 'adoption',
            'positive', 'optimistic', 'confidence', 'institutional', 'buying'
        ]
        
        # Bearish keywords
        bearish_keywords = [
            'bear', 'bearish', 'fall', 'falling', 'drop', 'dropping', 'decline',
            'declining', 'crash', 'crashing', 'dump', 'dumping', 'sell', 'selling',
            'down', 'lower', 'decrease', 'decreasing', 'loss', 'losses', 'fear',
            'panic', 'negative', 'pessimistic', 'concern', 'worried', 'risk'
        ]
        
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        
        total_keywords = bullish_count + bearish_count
        
        if total_keywords == 0:
            return {
                'sentiment': 'neutral',
                'score': 50,
                'confidence': 0.3
            }
        
        # Calculate sentiment score (0-100)
        bullish_ratio = bullish_count / total_keywords
        score = int(bullish_ratio * 100)
        
        # Determine sentiment label
        if score >= 60:
            sentiment = 'bullish'
        elif score <= 40:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on keyword count
        confidence = min(0.3 + (total_keywords * 0.1), 0.9)
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': round(confidence, 2)
        }
    
    def analyze_with_model(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment and score
        """
        if not self.use_transformers or not self.sentiment_model:
            return self.analyze_with_keywords(text)
        
        try:
            # Truncate text if too long
            text = text[:512]
            
            result = self.sentiment_model(text)[0]
            label = result['label'].lower()
            confidence = result['score']
            
            # Map FinBERT labels to our format
            if label == 'positive' or label == 'bullish':
                sentiment = 'bullish'
                score = int(50 + (confidence * 50))
            elif label == 'negative' or label == 'bearish':
                sentiment = 'bearish'
                score = int(50 - (confidence * 50))
            else:
                sentiment = 'neutral'
                score = 50
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': round(confidence, 2)
            }
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self.analyze_with_keywords(text)
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze a single article
        
        Args:
            article: Article dictionary with title and description
            
        Returns:
            Article dictionary with sentiment analysis added
        """
        text = f"{article.get('title', '')} {article.get('description', '')}"
        
        sentiment_result = self.analyze_with_model(text)
        
        article['sentiment'] = sentiment_result['sentiment']
        article['sentiment_score'] = sentiment_result['score']
        article['sentiment_confidence'] = sentiment_result['confidence']
        
        return article
    
    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze multiple articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with sentiment analysis
        """
        print(f"\nAnalyzing sentiment for {len(articles)} articles...")
        
        analyzed_articles = []
        
        for article in articles:
            analyzed_article = self.analyze_article(article)
            analyzed_articles.append(analyzed_article)
        
        # Calculate statistics
        sentiments = [a['sentiment'] for a in analyzed_articles]
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        neutral_count = sentiments.count('neutral')
        
        print(f"\nSentiment Distribution:")
        print(f"  Bullish: {bullish_count} ({bullish_count/len(articles)*100:.1f}%)")
        print(f"  Bearish: {bearish_count} ({bearish_count/len(articles)*100:.1f}%)")
        print(f"  Neutral: {neutral_count} ({neutral_count/len(articles)*100:.1f}%)")
        
        return analyzed_articles
    
    def save_analyzed_articles(self, articles: List[Dict], output_path: str):
        """
        Save analyzed articles to JSON file
        
        Args:
            articles: List of analyzed articles
            output_path: Path to save the file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analyzed_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'articles': articles
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved analyzed articles to {output_path}")

def main():
    """Main function for testing"""
    analyzer = SentimentAnalyzer()
    
    # Test with sample articles
    sample_articles = [
        {
            'title': 'Bitcoin surges to new all-time high as institutions buy',
            'description': 'Major rally in cryptocurrency markets'
        },
        {
            'title': 'Crypto market crashes amid regulatory concerns',
            'description': 'Bitcoin drops sharply on negative news'
        }
    ]
    
    analyzed = analyzer.analyze_articles(sample_articles)
    
    for article in analyzed:
        print(f"\nTitle: {article['title']}")
        print(f"Sentiment: {article['sentiment']} (Score: {article['sentiment_score']}, Confidence: {article['sentiment_confidence']})")

if __name__ == "__main__":
    main()
