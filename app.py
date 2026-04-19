import streamlit as st
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data_collector import DataCollector
from sentiment_model import SentimentAnalyzer
from predictor import StockPredictor

# Page config
st.set_page_config(page_title="Fin-Predictor", layout="wide", initial_sidebar_state="expanded")

# Caching models to prevent reloading
@st.cache_resource
def load_sentiment_model():
    return SentimentAnalyzer()

@st.cache_resource
def load_predictor_model(model_type, mode):
    return StockPredictor(model_type=model_type, mode=mode)

st.title("📈 Financial News Sentiment & Stock Predictor")
st.markdown("Predict tomorrow's stock direction based on today's technicals and news sentiment.")

# Sidebar
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")

st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Model Type", ["Random Forest", "XGBoost"])
mode = st.sidebar.radio("Prediction Mode", ["Classification (Up/Down)", "Regression (Price Change %)"])
internal_mode = "Classification" if "Classification" in mode else "Regression"

end_date_input = datetime.date.today()
start_date_input = end_date_input - datetime.timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", start_date_input)
end_date = st.sidebar.date_input("End Date", end_date_input)
run_btn = st.sidebar.button("Run Pipeline")

if run_btn:
    with st.spinner("Fetching Data and Analyzing..."):
        sa = load_sentiment_model()
        sp = load_predictor_model(model_type, internal_mode)
        
        # 1. Collect Data
        dc = DataCollector(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        stock_df = dc.fetch_stock_data()
        news_df = dc.fetch_news()
        
        if stock_df.empty:
            st.error(f"No stock data found for '{ticker}'. Please try another ticker.")
            st.stop()
            
        # 2. Analyze Sentiment
        detailed_news, agg_sentiment = sa.analyze_news(news_df)
        
        # 3. Prepare and Train Predictor
        merged_df = sp.prepare_data(stock_df, agg_sentiment)
        metrics, feature_importances, prediction, final_df = sp.train_and_evaluate(merged_df)
        
    st.success("Analysis Complete!")
    
    # -------- UI Panels --------
    
    # 1. Prediction Panel
    st.markdown("## 🔮 Prediction for Tomorrow")
    if prediction:
        if internal_mode == "Classification":
            dir_text = "UP 🟢" if prediction['Prediction'] == 1 else "DOWN 🔴"
            conf = prediction['Confidence'] * 100
            
            st.markdown(f"""
            <div class="content-box" style="text-align: center;">
                <h3 style="margin-bottom: 0px">Predicted Direction: {dir_text}</h3>
                <p style="font-size: 1.2rem; color: #8B949E;">Confidence Score: <b>{conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            pct_change = prediction['Prediction']
            dir_text = f"UP (+{pct_change:.2f}%) 🟢" if pct_change > 0 else f"DOWN ({pct_change:.2f}%) 🔴"
            
            st.markdown(f"""
            <div class="content-box" style="text-align: center;">
                <h3 style="margin-bottom: 0px">Predicted Price Move: {dir_text}</h3>
                <p style="font-size: 1.2rem; color: #8B949E;">Estimated from today's closing price</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Not enough data to train the model and make a prediction.")

    # 2. Charts
    st.markdown(f"## 📊 {ticker} Historical Performance & Sentiment")
    
    if not final_df.empty:
        # Plotly dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Stock Price
        fig.add_trace(
            go.Scatter(x=final_df['Date'], y=final_df['Close'], name="Close Price", line=dict(color='#58A6FF')),
            secondary_y=False,
        )
        # Moving Averages
        fig.add_trace(
            go.Scatter(x=final_df['Date'], y=final_df['SMA_20'], name="SMA 20", line=dict(color='#8B949E', dash='dot')),
            secondary_y=False,
        )
        
        # Positive Sentiment
        fig.add_trace(
            go.Bar(x=final_df['Date'], y=final_df['Sentiment_Pos'], name="Positive Sentiment", opacity=0.3, marker_color='#2EA043'),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text=f"{ticker} Price vs Next-Day Sentiment Proportion",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        fig.update_yaxes(title_text="Stock Price", secondary_y=False, showgrid=True, gridcolor='#30363D')
        fig.update_yaxes(title_text="Positive Sentiment Ratio", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # 3. Model Evaluation metrics
    st.markdown("## ⚙️ Model Evaluation")
    if metrics is not None:
        if internal_mode == "Classification":
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['Accuracy']*100:.1f}%")
            col2.metric("F1 Score", f"{metrics['F1_Score']:.2f}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.3f}%")
            col2.metric("Root Mean Sq. Error (RMSE)", f"{metrics['RMSE']:.3f}%")
        
        # Feature Importance
        st.markdown("### Feature Importance")
        fig_imp = go.Figure(go.Bar(
            x=feature_importances['Importance'],
            y=feature_importances['Feature'],
            orientation='h',
            marker_color='#58A6FF'
        ))
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'), height=300)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # 4. Recent News Feed
    st.markdown("## 📰 Recent News Feed")
    if not detailed_news.empty:
        # Show last 10 news headlines
        recent_news = detailed_news.sort_values(by='Date', ascending=False).head(10)
        for _, row in recent_news.iterrows():
            sentiment = row['Sentiment_Label']
            # Color coding
            if sentiment == 'positive':
                color_class = 'sent-pos'
            elif sentiment == 'negative':
                color_class = 'sent-neg'
            else:
                color_class = 'sent-neu'
            
            st.markdown(f"""
            <div class="content-box">
                <p style="margin: 0; font-size: 0.9em; color: #8B949E;">{row['Date']}</p>
                <h4 style="margin: 5px 0 5px 0;">{row['title']}</h4>
                <p style="margin: 0;">{row['summary']}</p>
                <p style="margin-top: 10px; font-size: 0.9em;">
                    Sentiment: <span class="{color_class}">{sentiment.upper()} ({row['Sentiment_Score']:.2f})</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No detailed news found for this period.")
    
    # End of UI    
