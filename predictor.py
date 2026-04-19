import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
import numpy as np

class StockPredictor:
    def __init__(self, model_type="Random Forest", mode="Classification"):
        self.model_type = model_type
        self.mode = mode
        
        if self.mode == "Classification":
            if self.model_type == "Random Forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            else:
                self.model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        else: # Regression
            if self.model_type == "Random Forest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                self.model = XGBRegressor(n_estimators=100, random_state=42)
                
        self.features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'Sentiment_Pos', 'Sentiment_Neg', 'Sentiment_Neu']

    def prepare_data(self, stock_df, sentiment_df):
        """
        Merges stock data and daily sentiment data.
        Fills missing sentiment with neutral scores.
        """
        # Merge on Date (Left join to keep all stock days)
        merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
        
        # Fill missing sentiment with neutral
        merged_df['Sentiment_Pos'].fillna(0.0, inplace=True)
        merged_df['Sentiment_Neg'].fillna(0.0, inplace=True)
        merged_df['Sentiment_Neu'].fillna(1.0, inplace=True)
        
        return merged_df

    def train_and_evaluate(self, merged_df):
        """
        Trains the model and evaluates it.
        Returns metrics, feature importances, and next day prediction.
        """
        target_col = 'Target' if self.mode == 'Classification' else 'Target_Reg'
        # The last row has NaN for target because it's shifted (tomorrow's data is unknown).
        # We will use all rows except the last one for training/evaluation
        train_df = merged_df.dropna(subset=[target_col])
        prediction_row = merged_df.iloc[-1:]
        
        if len(train_df) < 50:
            print("Not enough data to train a reliable model.")
            return None, None, None, None
            
        X = train_df[self.features]
        y = train_df[target_col]
        
        # Time-series split: 80% train, 20% test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        
        if self.mode == "Classification":
            metrics = {
                'Accuracy': accuracy_score(y_test, preds),
                'F1_Score': f1_score(y_test, preds),
                'Confusion_Matrix': confusion_matrix(y_test, preds).tolist()
            }
        else:
            metrics = {
                'MAE': mean_absolute_error(y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(y_test, preds))
            }
        
        feature_importances = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Predict tomorrow
        X_tomorrow = prediction_row[self.features]
        next_day_pred = float(self.model.predict(X_tomorrow)[0])
        
        prediction = {
            'Prediction': next_day_pred,
            'Date': prediction_row['Date'].values[0]
        }
        
        if self.mode == "Classification":
            prediction['Prediction'] = int(prediction['Prediction'])
            prediction['Confidence'] = float(self.model.predict_proba(X_tomorrow)[0].max())
            
        return metrics, feature_importances, prediction, merged_df

if __name__ == "__main__":
    pass
