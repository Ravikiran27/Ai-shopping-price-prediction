"""
Time-series price prediction using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

class TimeSeriesPricePredictor:
    """Time-series price prediction using historical data"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, price_col: str = 'price', lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Create lagged price features"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        for product_id in df['product_id'].unique():
            mask = df['product_id'] == product_id
            product_data = df[mask].copy()
            
            for lag in lags:
                df.loc[mask, f'price_lag_{lag}'] = product_data[price_col].shift(lag)
                df.loc[mask, f'price_change_{lag}'] = product_data[price_col].pct_change(lag)
            
            # Rolling statistics
            df.loc[mask, 'price_ma_7'] = product_data[price_col].rolling(7, min_periods=1).mean()
            df.loc[mask, 'price_ma_30'] = product_data[price_col].rolling(30, min_periods=1).mean()
            df.loc[mask, 'price_std_7'] = product_data[price_col].rolling(7, min_periods=1).std()
            df.loc[mask, 'price_volatility'] = product_data[price_col].rolling(7, min_periods=1).std() / product_data[price_col].rolling(7, min_periods=1).mean()
        
        return df
    
    def prepare_training_data(self, price_history_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for time-series training"""
        # Merge with product information
        df = price_history_df.merge(products_df, left_on='product_id', right_on='id', suffixes=('', '_product'))
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Product features
        df['category_encoded'] = pd.Categorical(df['category']).codes
        df['brand_encoded'] = pd.Categorical(df['brand']).codes
        df['rating_filled'] = df['rating'].fillna(df['rating'].mean())
        df['num_reviews_filled'] = df['num_reviews'].fillna(0)
        
        # Select features
        feature_cols = [
            'year', 'month', 'day', 'day_of_week', 'week_of_year', 'quarter',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'price_lag_1', 'price_lag_7', 'price_lag_30',
            'price_change_1', 'price_change_7', 'price_change_30',
            'price_ma_7', 'price_ma_30', 'price_std_7', 'price_volatility',
            'category_encoded', 'brand_encoded', 'rating_filled', 'num_reviews_filled'
        ]
        
        # Remove rows with NaN values (mainly from lag features)
        df = df.dropna(subset=feature_cols + ['price'])
        
        self.feature_columns = feature_cols
        X = df[feature_cols]
        y = df['price']
        
        return X, y, df
    
    def train(self, price_history_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, float]:
        """Train the time-series model"""
        try:
            X, y, full_df = self.prepare_training_data(price_history_df, products_df)
            
            if len(X) < 50:
                raise ValueError("Insufficient data for time-series training. Need at least 50 historical price points.")
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            self.is_trained = True
            logging.info(f"Time-series model trained successfully. RMSE: {metrics['rmse']:.2f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error training time-series model: {str(e)}")
            raise e
    
    def predict_future_prices(self, product_id: int, days_ahead: int, 
                            last_prices: pd.DataFrame, product_info: Dict) -> pd.DataFrame:
        """Predict future prices for a product"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_date = datetime.now()
        
        # Get the last known prices for lag features
        last_prices_sorted = last_prices.sort_values('timestamp')
        
        for day in range(1, days_ahead + 1):
            future_date = current_date + timedelta(days=day)
            
            # Create features for future date
            features = {
                'year': future_date.year,
                'month': future_date.month,
                'day': future_date.day,
                'day_of_week': future_date.weekday(),
                'week_of_year': future_date.isocalendar()[1],
                'quarter': (future_date.month - 1) // 3 + 1,
                'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                'day_sin': np.sin(2 * np.pi * future_date.weekday() / 7),
                'day_cos': np.cos(2 * np.pi * future_date.weekday() / 7),
            }
            
            # Lag features (use last known prices or previous predictions)
            if len(last_prices_sorted) > 0:
                last_price = last_prices_sorted.iloc[-1]['price']
                features['price_lag_1'] = last_price
                
                if len(last_prices_sorted) >= 7:
                    features['price_lag_7'] = last_prices_sorted.iloc[-7]['price']
                    features['price_ma_7'] = last_prices_sorted.tail(7)['price'].mean()
                    features['price_std_7'] = last_prices_sorted.tail(7)['price'].std()
                    features['price_volatility'] = features['price_std_7'] / features['price_ma_7']
                else:
                    features['price_lag_7'] = last_price
                    features['price_ma_7'] = last_price
                    features['price_std_7'] = 0
                    features['price_volatility'] = 0
                
                if len(last_prices_sorted) >= 30:
                    features['price_lag_30'] = last_prices_sorted.iloc[-30]['price']
                    features['price_ma_30'] = last_prices_sorted.tail(30)['price'].mean()
                else:
                    features['price_lag_30'] = last_price
                    features['price_ma_30'] = last_price
                
                # Price changes
                features['price_change_1'] = 0  # Will be updated with actual changes
                features['price_change_7'] = 0
                features['price_change_30'] = 0
            else:
                # Default values if no history
                for lag in [1, 7, 30]:
                    features[f'price_lag_{lag}'] = product_info.get('price', 100)
                    features[f'price_change_{lag}'] = 0
                features['price_ma_7'] = product_info.get('price', 100)
                features['price_ma_30'] = product_info.get('price', 100)
                features['price_std_7'] = 0
                features['price_volatility'] = 0
            
            # Product features
            features['category_encoded'] = 0  # Will need proper encoding
            features['brand_encoded'] = 0     # Will need proper encoding
            features['rating_filled'] = product_info.get('rating', 4.0)
            features['num_reviews_filled'] = product_info.get('num_reviews', 0)
            
            # Create feature vector
            feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            predicted_price = self.model.predict(feature_vector_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'predicted_price': predicted_price,
                'confidence': 0.8  # Placeholder confidence score
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']