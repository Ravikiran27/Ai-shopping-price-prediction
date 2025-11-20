"""
Recommendation system for e-commerce products
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

from config.config import MODEL_CONFIG
from data.preprocessor import DataPreprocessor

class RecommendationSystem:
    """Hybrid recommendation system combining collaborative and content-based filtering"""
    
    def __init__(self):
        self.collaborative_model = None
        self.content_vectorizer = None
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.product_features = None
        self.user_features = None
        self.preprocessor = DataPreprocessor()
        
        # Model parameters
        self.cf_params = MODEL_CONFIG['recommendation']['collaborative_filtering']
        self.cb_params = MODEL_CONFIG['recommendation']['content_based']
        self.hybrid_params = MODEL_CONFIG['recommendation']['hybrid']
    
    def prepare_data(self, interactions_df: pd.DataFrame, 
                    products_df: pd.DataFrame, 
                    users_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for recommendation system"""
        logging.info("Preprocessing data for recommendation system...")
        
        # Preprocess data using the preprocessor
        preprocessed_data = self.preprocessor.preprocess_for_recommendations(
            interactions_df, products_df, users_df
        )
        
        self.user_item_matrix = preprocessed_data['user_item_matrix']
        self.product_features = preprocessed_data['product_features']
        self.user_features = preprocessed_data['user_features']
        self.item_similarity_matrix = preprocessed_data['item_similarity']
        
        logging.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
        logging.info(f"Product features shape: {self.product_features.shape}")
        
        return preprocessed_data
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering model using SVD"""
        logging.info("Training collaborative filtering model...")
        
        # Use Truncated SVD for matrix factorization
        n_components = min(50, min(self.user_item_matrix.shape) - 1)
        self.collaborative_model = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )
        
        # Fit on user-item matrix
        self.collaborative_model.fit(self.user_item_matrix)
        
        logging.info(f"Collaborative filtering model trained with {n_components} components")
    
    def train_content_based(self, products_df: pd.DataFrame):
        """Train content-based filtering using TF-IDF"""
        logging.info("Training content-based filtering model...")
        
        # Create text features for products
        text_features = []
        for _, product in products_df.iterrows():
            text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')} {product.get('brand', '')}"
            text_features.append(text)
        
        # Initialize and fit TF-IDF vectorizer
        self.content_vectorizer = TfidfVectorizer(
            max_features=self.cb_params['max_features'],
            ngram_range=self.cb_params['ngram_range'],
            stop_words=self.cb_params['stop_words']
        )
        
        # Fit and transform
        content_matrix = self.content_vectorizer.fit_transform(text_features)
        
        # Calculate content-based similarity
        self.content_similarity = cosine_similarity(content_matrix)
        
        logging.info("Content-based filtering model trained")
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations using collaborative filtering"""
        if self.collaborative_model is None:
            raise ValueError("Collaborative filtering model not trained")
        
        if user_id not in self.user_item_matrix.index:
            # Handle cold start - return popular items
            return self._get_popular_items(n_recommendations)
        
        # Get user's latent factors
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_factors = self.collaborative_model.transform(self.user_item_matrix.iloc[[user_idx]])
        
        # Get item factors
        item_factors = self.collaborative_model.components_
        
        # Calculate predicted ratings
        predicted_ratings = np.dot(user_factors, item_factors)[0]
        
        # Get user's already interacted items
        user_interactions = self.user_item_matrix.iloc[user_idx]
        already_interacted = user_interactions[user_interactions > 0].index.tolist()
        
        # Get recommendations (excluding already interacted items)
        item_scores = list(zip(self.user_item_matrix.columns, predicted_ratings))
        item_scores = [(item_id, score) for item_id, score in item_scores 
                      if item_id not in already_interacted]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in item_scores[:n_recommendations]:
            recommendations.append({
                'product_id': item_id,
                'score': float(score),
                'reason': 'Users with similar preferences also liked this'
            })
        
        return recommendations
    
    def get_content_based_recommendations(self, user_id: int = None, 
                                        product_id: int = None, 
                                        n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations using content-based filtering"""
        if self.content_similarity is None:
            raise ValueError("Content-based model not trained")
        
        if product_id is not None:
            # Item-based recommendations
            return self._get_similar_items(product_id, n_recommendations)
        
        elif user_id is not None:
            # User-based content recommendations
            return self._get_user_content_recommendations(user_id, n_recommendations)
        
        else:
            raise ValueError("Either user_id or product_id must be provided")
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining collaborative and content-based"""
        # Get collaborative recommendations
        try:
            collaborative_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        except:
            collaborative_recs = []
        
        # Get content-based recommendations
        try:
            content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
        except:
            content_recs = []
        
        # Combine recommendations with weights
        cf_weight = self.hybrid_params['collaborative_weight']
        cb_weight = self.hybrid_params['content_weight']
        
        # Create combined scores
        combined_scores = {}
        
        # Add collaborative filtering scores
        for rec in collaborative_recs:
            product_id = rec['product_id']
            combined_scores[product_id] = combined_scores.get(product_id, 0) + cf_weight * rec['score']
        
        # Add content-based scores
        for rec in content_recs:
            product_id = rec['product_id']
            combined_scores[product_id] = combined_scores.get(product_id, 0) + cb_weight * rec['score']
        
        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product_id, score in sorted_items[:n_recommendations]:
            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'reason': 'Based on your preferences and similar products'
            })
        
        return recommendations
    
    def _get_similar_items(self, product_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get items similar to a given product"""
        if product_id >= len(self.content_similarity):
            return []
        
        # Get similarity scores for the product
        similarity_scores = self.content_similarity[product_id]
        
        # Get indices of most similar items (excluding the item itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'product_id': idx,
                'score': float(similarity_scores[idx]),
                'reason': 'Similar to your viewed product'
            })
        
        return recommendations
    
    def _get_user_content_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get content-based recommendations for a user based on their interaction history"""
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_items(n_recommendations)
        
        # Get user's interaction history
        user_interactions = self.user_item_matrix.loc[user_id]
        liked_items = user_interactions[user_interactions > 0].index.tolist()
        
        if not liked_items:
            return self._get_popular_items(n_recommendations)
        
        # Calculate average similarity to liked items
        item_scores = {}
        
        for product_id in range(len(self.content_similarity)):
            if product_id in liked_items:
                continue
            
            # Calculate average similarity to user's liked items
            similarities = [self.content_similarity[product_id][liked_item] 
                          for liked_item in liked_items if liked_item < len(self.content_similarity)]
            
            if similarities:
                avg_similarity = np.mean(similarities)
                item_scores[product_id] = avg_similarity
        
        # Sort by similarity score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product_id, score in sorted_items[:n_recommendations]:
            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'reason': 'Similar to items you liked before'
            })
        
        return recommendations
    
    def _get_popular_items(self, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get popular items for cold start problem"""
        if self.user_item_matrix is None:
            return []
        
        # Calculate item popularity (sum of interactions)
        item_popularity = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
        
        recommendations = []
        for product_id, popularity in item_popularity.head(n_recommendations).items():
            recommendations.append({
                'product_id': product_id,
                'score': float(popularity),
                'reason': 'Popular among all users'
            })
        
        return recommendations
    
    def get_user_recommendations(self, user_id: int, 
                               recommendation_type: str = 'hybrid',
                               n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations for a user with specified type"""
        
        if recommendation_type == 'collaborative':
            return self.get_collaborative_recommendations(user_id, n_recommendations)
        elif recommendation_type == 'content':
            return self.get_content_based_recommendations(user_id=user_id, n_recommendations=n_recommendations)
        elif recommendation_type == 'hybrid':
            return self.get_hybrid_recommendations(user_id, n_recommendations)
        else:
            raise ValueError(f"Unknown recommendation type: {recommendation_type}")
    
    def get_item_recommendations(self, product_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get items similar to a given product"""
        return self.get_content_based_recommendations(product_id=product_id, n_recommendations=n_recommendations)
    
    def evaluate_recommendations(self, test_interactions: pd.DataFrame, 
                               recommendation_type: str = 'hybrid',
                               k: int = 10) -> Dict[str, float]:
        """Evaluate recommendation system using precision@k and recall@k"""
        
        precisions = []
        recalls = []
        
        # Group test interactions by user
        user_test_items = test_interactions.groupby('user_id')['product_id'].apply(list).to_dict()
        
        for user_id, true_items in user_test_items.items():
            try:
                # Get recommendations
                recommendations = self.get_user_recommendations(
                    user_id, recommendation_type, k
                )
                
                recommended_items = [rec['product_id'] for rec in recommendations]
                
                # Calculate precision and recall
                if recommended_items and true_items:
                    intersection = set(recommended_items) & set(true_items)
                    
                    precision = len(intersection) / len(recommended_items) if recommended_items else 0
                    recall = len(intersection) / len(true_items) if true_items else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    
            except Exception as e:
                logging.warning(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall,
            'f1_score': f1_score,
            'users_evaluated': len(precisions)
        }
    
    def get_recommendation_explanation(self, user_id: int, product_id: int) -> Dict[str, Any]:
        """Get explanation for why a product was recommended to a user"""
        explanations = []
        
        # Check collaborative filtering explanation
        if self.collaborative_model is not None and user_id in self.user_item_matrix.index:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_factors = self.collaborative_model.transform(self.user_item_matrix.iloc[[user_idx]])
            
            # Find similar users
            all_user_factors = self.collaborative_model.transform(self.user_item_matrix)
            user_similarities = cosine_similarity(user_factors, all_user_factors)[0]
            
            similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
            similar_user_ids = [self.user_item_matrix.index[idx] for idx in similar_users]
            
            explanations.append({
                'type': 'collaborative',
                'explanation': f"Users similar to you (IDs: {similar_user_ids[:3]}) also liked this product",
                'confidence': float(np.max(user_similarities[similar_users]))
            })
        
        # Check content-based explanation
        if self.content_similarity is not None:
            user_interactions = self.user_item_matrix.loc[user_id] if user_id in self.user_item_matrix.index else pd.Series()
            liked_items = user_interactions[user_interactions > 0].index.tolist()
            
            if liked_items and product_id < len(self.content_similarity):
                similarities_to_liked = [
                    self.content_similarity[product_id][item] 
                    for item in liked_items if item < len(self.content_similarity)
                ]
                
                if similarities_to_liked:
                    max_similarity = max(similarities_to_liked)
                    most_similar_item = liked_items[similarities_to_liked.index(max_similarity)]
                    
                    explanations.append({
                        'type': 'content',
                        'explanation': f"This product is similar to item {most_similar_item} that you liked",
                        'confidence': float(max_similarity)
                    })
        
        return {
            'user_id': user_id,
            'product_id': product_id,
            'explanations': explanations
        }
    
    def save_models(self, models_dir: str = None):
        """Save recommendation models"""
        if models_dir is None:
            models_dir = MODEL_CONFIG['models_dir']
        
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save collaborative filtering model
        if self.collaborative_model is not None:
            joblib.dump(self.collaborative_model, models_dir / "recommendation_collaborative.joblib")
        
        # Save content vectorizer
        if self.content_vectorizer is not None:
            joblib.dump(self.content_vectorizer, models_dir / "recommendation_content_vectorizer.joblib")
        
        # Save similarity matrices and data
        matrices_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_similarity_matrix': self.item_similarity_matrix,
            'content_similarity': getattr(self, 'content_similarity', None),
            'product_features': self.product_features,
            'user_features': self.user_features
        }
        joblib.dump(matrices_data, models_dir / "recommendation_matrices.joblib")
        
        # Save preprocessor
        preprocessor_path = models_dir / "recommendation_preprocessor.joblib"
        self.preprocessor.save_preprocessors(preprocessor_path)
        
        logging.info("Recommendation models saved successfully")
    
    def load_models(self, models_dir: str = None):
        """Load recommendation models"""
        if models_dir is None:
            models_dir = MODEL_CONFIG['models_dir']
        
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            logging.warning(f"Models directory {models_dir} does not exist")
            return
        
        # Load collaborative filtering model
        cf_path = models_dir / "recommendation_collaborative.joblib"
        if cf_path.exists():
            self.collaborative_model = joblib.load(cf_path)
        
        # Load content vectorizer
        cv_path = models_dir / "recommendation_content_vectorizer.joblib"
        if cv_path.exists():
            self.content_vectorizer = joblib.load(cv_path)
        
        # Load matrices
        matrices_path = models_dir / "recommendation_matrices.joblib"
        if matrices_path.exists():
            matrices_data = joblib.load(matrices_path)
            self.user_item_matrix = matrices_data.get('user_item_matrix')
            self.item_similarity_matrix = matrices_data.get('item_similarity_matrix')
            self.content_similarity = matrices_data.get('content_similarity')
            self.product_features = matrices_data.get('product_features')
            self.user_features = matrices_data.get('user_features')
        
        # Load preprocessor
        preprocessor_path = models_dir / "recommendation_preprocessor.joblib"
        if preprocessor_path.exists():
            self.preprocessor.load_preprocessors(preprocessor_path)
        
        logging.info("Recommendation models loaded successfully")