"""
Data preprocessing utilities for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import json
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

class DataPreprocessor:
    """Handles data preprocessing for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_columns = []
    
    def preprocess_products_for_price_prediction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess product data for price prediction"""
        processed_df = df.copy()
        preprocessing_info = {}
        
        # Handle missing values
        processed_df['brand'] = processed_df['brand'].fillna('Unknown')
        processed_df['description'] = processed_df['description'].fillna('')
        processed_df['rating'] = processed_df['rating'].fillna(processed_df['rating'].mean())
        processed_df['num_reviews'] = processed_df['num_reviews'].fillna(0)
        
        # Extract features from features JSON column
        if 'features' in processed_df.columns:
            features_data = processed_df['features'].apply(self._parse_features)
            feature_df = pd.json_normalize(features_data)
            processed_df = pd.concat([processed_df, feature_df], axis=1)
        
        # Text features from product name and description
        processed_df['name_length'] = processed_df['name'].str.len()
        processed_df['description_length'] = processed_df['description'].str.len()
        processed_df['word_count'] = processed_df['description'].str.split().str.len()
        
        # Extract sentiment from description
        processed_df['description_sentiment'] = processed_df['description'].apply(self._get_sentiment)
        
        # Extract numeric features from text
        processed_df = self._extract_numeric_features(processed_df)
        
        # Encode categorical variables
        categorical_columns = ['category', 'brand']
        for col in categorical_columns:
            if col in processed_df.columns:
                processed_df[f'{col}_encoded'] = self._encode_categorical(col, processed_df[col])
        
        # Create price bins for analysis
        processed_df['price_category'] = pd.cut(processed_df['price'], 
                                              bins=5, 
                                              labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Feature engineering
        processed_df['rating_reviews_ratio'] = processed_df['rating'] * np.log1p(processed_df['num_reviews'])
        processed_df['price_per_rating'] = processed_df['price'] / (processed_df['rating'] + 1e-6)
        
        # Store preprocessing info
        preprocessing_info = {
            'feature_columns': list(processed_df.columns),
            'categorical_encoders': {k: v for k, v in self.encoders.items()},
            'scalers': {k: v for k, v in self.scalers.items()},
            'shape': processed_df.shape
        }
        
        return processed_df, preprocessing_info
    
    def preprocess_for_recommendations(self, interactions_df: pd.DataFrame, 
                                     products_df: pd.DataFrame, 
                                     users_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Preprocess data for recommendation system"""
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating',
            fill_value=0
        )
        
        # Product features for content-based filtering
        product_features = self._create_product_feature_matrix(products_df)
        
        # User features
        user_features = self._create_user_feature_matrix(users_df, interactions_df)
        
        # Item similarity matrix
        item_similarity = self._calculate_item_similarity(product_features)
        
        return {
            'user_item_matrix': user_item_matrix,
            'product_features': product_features,
            'user_features': user_features,
            'item_similarity': item_similarity
        }
    
    def _parse_features(self, features_str: str) -> Dict[str, Any]:
        """Parse features JSON string"""
        try:
            if pd.isna(features_str) or features_str == '':
                return {}
            return json.loads(features_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _get_sentiment(self, text: str) -> float:
        """Get sentiment score from text"""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric features from text columns"""
        processed_df = df.copy()
        
        # Extract numbers from product names and descriptions
        for col in ['name', 'description']:
            if col in processed_df.columns:
                # Extract sizes (like 13", 15", etc.)
                sizes = processed_df[col].str.extract(r'(\d+)(?:"|inch|inches)', expand=False)
                processed_df[f'{col}_size'] = pd.to_numeric(sizes, errors='coerce').fillna(0)
                
                # Extract storage/memory (like 256GB, 8GB RAM)
                storage = processed_df[col].str.extract(r'(\d+)(?:GB|TB|MB)', expand=False)
                processed_df[f'{col}_storage'] = pd.to_numeric(storage, errors='coerce').fillna(0)
                
                # Extract years
                years = processed_df[col].str.extract(r'(20\d{2})', expand=False)
                processed_df[f'{col}_year'] = pd.to_numeric(years, errors='coerce').fillna(0)
        
        return processed_df
    
    def _encode_categorical(self, column_name: str, series: pd.Series) -> pd.Series:
        """Encode categorical variables"""
        if column_name not in self.encoders:
            self.encoders[column_name] = LabelEncoder()
            return pd.Series(self.encoders[column_name].fit_transform(series.astype(str)))
        else:
            # Handle new categories during prediction
            known_categories = set(self.encoders[column_name].classes_)
            series_str = series.astype(str)
            unknown_mask = ~series_str.isin(known_categories)
            
            # Assign unknown categories to a default value
            series_processed = series_str.copy()
            if unknown_mask.any():
                series_processed[unknown_mask] = self.encoders[column_name].classes_[0]
            
            return pd.Series(self.encoders[column_name].transform(series_processed))
    
    def _create_product_feature_matrix(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for products"""
        features_df = products_df.copy()
        
        # Text features using TF-IDF
        text_features = (features_df['name'].fillna('') + ' ' + 
                        features_df['description'].fillna('') + ' ' + 
                        features_df['category'].fillna('') + ' ' + 
                        features_df['brand'].fillna(''))
        
        if 'product_text' not in self.vectorizers:
            self.vectorizers['product_text'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.vectorizers['product_text'].fit_transform(text_features)
        else:
            tfidf_matrix = self.vectorizers['product_text'].transform(text_features)
        
        # Convert to DataFrame
        feature_names = self.vectorizers['product_text'].get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=feature_names, 
                               index=features_df.index)
        
        # Add numeric features
        numeric_features = ['price', 'rating', 'num_reviews']
        for feature in numeric_features:
            if feature in features_df.columns:
                tfidf_df[feature] = features_df[feature].fillna(0)
        
        # Add categorical features (one-hot encoded)
        categorical_features = ['category', 'brand']
        for feature in categorical_features:
            if feature in features_df.columns:
                dummies = pd.get_dummies(features_df[feature], prefix=feature)
                tfidf_df = pd.concat([tfidf_df, dummies], axis=1)
        
        return tfidf_df
    
    def _create_user_feature_matrix(self, users_df: pd.DataFrame, 
                                   interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for users"""
        user_features = users_df.copy()
        
        # Parse preferences
        if 'preferences' in user_features.columns:
            prefs_data = user_features['preferences'].apply(self._parse_features)
            prefs_df = pd.json_normalize(prefs_data)
            user_features = pd.concat([user_features, prefs_df], axis=1)
        
        # Add interaction-based features
        user_stats = interactions_df.groupby('user_id').agg({
            'product_id': 'count',  # Total interactions
            'rating': ['mean', 'std'],  # Rating statistics
            'price_paid': ['mean', 'sum', 'std']  # Purchase statistics
        }).round(2)
        
        user_stats.columns = ['total_interactions', 'avg_rating', 'rating_std', 
                             'avg_purchase_price', 'total_spent', 'purchase_price_std']
        user_stats = user_stats.fillna(0)
        
        # Merge with user features
        user_features = user_features.set_index('id').join(user_stats, how='left').fillna(0)
        
        return user_features
    
    def _calculate_item_similarity(self, product_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate item-item similarity matrix"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Normalize features
        if 'product_similarity' not in self.scalers:
            self.scalers['product_similarity'] = StandardScaler()
            normalized_features = self.scalers['product_similarity'].fit_transform(product_features)
        else:
            normalized_features = self.scalers['product_similarity'].transform(product_features)
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(normalized_features)
        
        return pd.DataFrame(similarity_matrix, 
                          index=product_features.index, 
                          columns=product_features.index)
    
    def prepare_features_for_ml(self, df: pd.DataFrame, 
                               target_column: str = 'price',
                               feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML models"""
        
        if feature_columns is None:
            # Automatically select numeric columns
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
            
            # Remove ID columns and other irrelevant columns
            exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        if 'ml_features' not in self.scalers:
            self.scalers['ml_features'] = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scalers['ml_features'].fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scalers['ml_features'].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        self.feature_columns = feature_columns
        
        return X_scaled, y
    
    def inverse_transform_features(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features"""
        if 'ml_features' in self.scalers:
            X_original = pd.DataFrame(
                self.scalers['ml_features'].inverse_transform(X_scaled),
                columns=X_scaled.columns,
                index=X_scaled.index
            )
            return X_original
        return X_scaled
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis"""
        return self.feature_columns
    
    def save_preprocessors(self, filepath: str):
        """Save preprocessors to file"""
        import joblib
        
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'vectorizers': self.vectorizers,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, filepath)
    
    def load_preprocessors(self, filepath: str):
        """Load preprocessors from file"""
        import joblib
        
        preprocessor_data = joblib.load(filepath)
        
        self.scalers = preprocessor_data.get('scalers', {})
        self.encoders = preprocessor_data.get('encoders', {})
        self.vectorizers = preprocessor_data.get('vectorizers', {})
        self.feature_columns = preprocessor_data.get('feature_columns', [])