# Financial News Sentiment & Stock Predictor

This is a complete end-to-end Python project that performs sentiment analysis on financial news headlines to predict stock market direction (up or down) for the next day.

## Features
- **Data Collection**: Retrieves historical stock prices using `yfinance` and financial news using `finnhub`. If no Finnhub API key is provided, the system intelligently falls back to realistic mock news data allowing the app to run perfectly out of the box. Computes critical technical indicators like RSI, SMA, and MACD.
- **Sentiment Analysis**: Uses the pre-trained `ProsusAI/finbert` from Hugging Face to evaluate headlines specifically fine-tuned for financial sentiment. It extracts positive, negative, and neutral scores.
- **Machine Learning**: A RandomForestClassifier trained on a combination of technical indicators and sentiment scores predicting whether the stock will close higher tomorrow. 
- **Modern UI**: A sleek, dark-themed Streamlit dashboard providing interactive Plotly charts, model evaluation metrics, and a clean news feed.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.8+ installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   The project requires an API key to fetch real financial news.
   - Go to [Finnhub.io](https://finnhub.io/) and generate a free API key.
   - Rename `.env.example` to `.env`.
   - Add your API key: `FINNHUB_API_KEY=your_key_here`.

   *Note: If you do not provide an API key, the app relies on internally generated mock news to demonstrate functionality.*

3. **Run the Application**
   Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
   *The very first time you click "Run Pipeline" in the app, it will download the ProsusAI/finbert model (approximately 400MB).*

## Project Structure
- `data_collector.py`: Handles fetching data from `yfinance` and `finnhub`. Cleans data and calculates technical indicators.
- `sentiment_model.py`: Handles the NLP layer using Hugging Face pipelines to assign sentiment to news.
- `predictor.py`: The Machine Learning layer that trains a Random Forest model on the combined data and makes future predictions.
- `app.py`: The user interface built with Streamlit.

## Future Enhancements
- Expand to tick-level data.
- Integrate Twitter / X sentiment.
- Support deep learning models (LSTM) for the time-series predictions instead of Random Forest.
