import pandas as pd
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        """
        Initializes the FinBERT model for sentiment analysis.
        This model is specifically fine-tuned on financial texts.
        """
        print("Loading ProsusAI/finbert model... This might take a moment on first run.")
        self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1) # change device=0 for GPU

    def analyze_news(self, news_df):
        """
        Analyzes sentiment of the given news dataframe.
        Returns the news dataframe with sentiment labels, plus an aggregated daily dataset.
        """
        if news_df is None or news_df.empty:
            print("No news to analyze.")
            return pd.DataFrame(), pd.DataFrame(columns=['Date', 'Sentiment_Pos', 'Sentiment_Neg', 'Sentiment_Neu'])
        
        # Combine title and summary for richer text analysis
        texts = (news_df['title'].fillna('') + ". " + news_df['summary'].fillna('')).tolist()
        
        # Truncate to maximum standard token lengths if text is too long (basic string slicing)
        texts = [t[:512] for t in texts]
        
        print(f"Running FinBERT sentiment over {len(texts)} news articles...")
        # Note: In a production environment with huge data, process in batches.
        results = self.analyzer(texts)
        
        news_df['Sentiment_Label'] = [res['label'] for res in results]
        news_df['Sentiment_Score'] = [res['score'] for res in results]
        
        # Aggregate sentiment by Date
        daily_sentiment = []
        for date, group in news_df.groupby('Date'):
            pos_score = group[group['Sentiment_Label'] == 'positive']['Sentiment_Score'].sum()
            neg_score = group[group['Sentiment_Label'] == 'negative']['Sentiment_Score'].sum()
            neu_score = group[group['Sentiment_Label'] == 'neutral']['Sentiment_Score'].sum()
            
            total = pos_score + neg_score + neu_score
            if total == 0:
                pos, neg, neu = 0.0, 0.0, 1.0 # default to neutral
            else:
                pos = pos_score / total
                neg = neg_score / total
                neu = neu_score / total
                
            daily_sentiment.append({
                'Date': date,
                'Sentiment_Pos': pos,
                'Sentiment_Neg': neg,
                'Sentiment_Neu': neu
            })
            
        agg_df = pd.DataFrame(daily_sentiment)
        return news_df, agg_df

if __name__ == "__main__":
    # Test Sentiment Analyzer
    dummy_news = pd.DataFrame({
        'Date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-01').date()],
        'title': ['Apple revenue beats estimates', 'Stock plummets due to regulatory concerns'],
        'summary': ['A massive quarter for Apple.', 'Fears of incoming bans.']
    })
    
    sa = SentimentAnalyzer()
    detailed_df, daily_df = sa.analyze_news(dummy_news)
    print("Detailed Sentiment:")
    print(detailed_df)
    print("\nDaily Aggregated Sentiment:")
    print(daily_df)
