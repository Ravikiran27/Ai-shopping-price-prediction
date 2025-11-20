"""
Price prediction models for e-commerce products
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

from config.config import MODEL_CONFIG
from data.preprocessor import DataPreprocessor

class PricePredictionModel:
    """Handles price prediction using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.preprocessor = DataPreprocessor()
        self.feature_importance = {}
        self.best_model_name = None
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=MODEL_CONFIG['price_prediction']['random_state'],
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=MODEL_CONFIG['price_prediction']['random_state']
            ),
            'linear_regression': LinearRegression()
        }
    
    def prepare_data(self, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        logging.info("Preprocessing data for price prediction...")
        
        # Preprocess products data
        processed_df, preprocessing_info = self.preprocessor.preprocess_products_for_price_prediction(products_df)
        
        # Prepare features and target
        X, y = self.preprocessor.prepare_features_for_ml(processed_df, target_column='price')
        
        logging.info(f"Prepared data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = None) -> Dict[str, Dict[str, float]]:
        """Train all models and evaluate performance"""
        
        if test_size is None:
            test_size = MODEL_CONFIG['price_prediction']['test_size']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=MODEL_CONFIG['price_prediction']['random_state']
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=MODEL_CONFIG['price_prediction']['cv_folds'],
                    scoring='neg_mean_squared_error'
                )
                
                # Store results
                results[model_name] = {
                    'train_rmse': train_metrics['rmse'],
                    'test_rmse': test_metrics['rmse'],
                    'train_mae': train_metrics['mae'],
                    'test_mae': test_metrics['mae'],
                    'train_r2': train_metrics['r2'],
                    'test_r2': test_metrics['r2'],
                    'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                    'cv_rmse_std': np.sqrt(cv_scores.std())
                }
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_names = self.preprocessor.get_feature_importance_names()
                    self.feature_importance[model_name] = dict(zip(
                        feature_names, 
                        model.feature_importances_
                    ))
                
                logging.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.3f}")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            self.best_model_name = min(valid_results.keys(), 
                                     key=lambda x: valid_results[x]['test_rmse'])
            logging.info(f"Best model: {self.best_model_name}")
        
        self.model_performance = results
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make price predictions"""
        if model_name is None:
            model_name = self.best_model_name or 'random_forest'
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_single_product(self, product_data: Dict[str, Any], 
                             model_name: str = None) -> Dict[str, Any]:
        """Predict price for a single product"""
        if model_name is None:
            model_name = self.best_model_name or 'random_forest'
        
        # Convert to DataFrame
        product_df = pd.DataFrame([product_data])
        
        # Preprocess
        processed_df, _ = self.preprocessor.preprocess_products_for_price_prediction(product_df)
        X, _ = self.preprocessor.prepare_features_for_ml(processed_df, target_column='price')
        
        # Predict
        prediction = self.predict(X, model_name)[0]
        
        # Get confidence interval (using ensemble predictions if available)
        confidence_interval = self._get_prediction_confidence(X, model_name)
        
        # Get feature contributions
        contributions = self._get_feature_contributions(X, model_name)
        
        return {
            'predicted_price': float(prediction),
            'confidence_interval': confidence_interval,
            'model_used': model_name,
            'feature_contributions': contributions,
            'model_performance': self.model_performance.get(model_name, {})
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _get_prediction_confidence(self, X: pd.DataFrame, model_name: str) -> Tuple[float, float]:
        """Get prediction confidence interval"""
        model = self.models[model_name]
        
        if model_name == 'random_forest':
            # Use tree predictions to estimate uncertainty
            predictions = np.array([tree.predict(X) for tree in model.estimators_])
            prediction_std = np.std(predictions, axis=0)[0]
            prediction_mean = np.mean(predictions, axis=0)[0]
            
            # 95% confidence interval
            ci_lower = prediction_mean - 1.96 * prediction_std
            ci_upper = prediction_mean + 1.96 * prediction_std
            
            return (float(ci_lower), float(ci_upper))
        
        else:
            # Simple approach: use model performance to estimate uncertainty
            test_rmse = self.model_performance.get(model_name, {}).get('test_rmse', 0)
            prediction = model.predict(X)[0]
            
            ci_lower = prediction - 1.96 * test_rmse
            ci_upper = prediction + 1.96 * test_rmse
            
            return (float(ci_lower), float(ci_upper))
    
    def _get_feature_contributions(self, X: pd.DataFrame, model_name: str) -> Dict[str, float]:
        """Get feature contributions to prediction"""
        if model_name not in self.feature_importance:
            return {}
        
        feature_names = self.preprocessor.get_feature_importance_names()
        if len(feature_names) != len(X.columns):
            return {}
        
        contributions = {}
        for i, feature_name in enumerate(feature_names):
            if i < len(X.columns):
                importance = self.feature_importance[model_name].get(feature_name, 0)
                feature_value = X.iloc[0, i]
                contributions[feature_name] = float(importance * feature_value)
        
        return contributions
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                           model_name: str = 'random_forest') -> Dict[str, Any]:
        """Tune hyperparameters for a specific model"""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        if model_name not in param_grids:
            logging.warning(f"No hyperparameter grid defined for {model_name}")
            return {}
        
        logging.info(f"Tuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logging.info(f"Best cross-validation score: {-grid_search.best_score_:.2f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models"""
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.model_performance.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Model': model_name,
                    'Test RMSE': metrics.get('test_rmse', 0),
                    'Test MAE': metrics.get('test_mae', 0),
                    'Test R²': metrics.get('test_r2', 0),
                    'CV RMSE Mean': metrics.get('cv_rmse_mean', 0),
                    'CV RMSE Std': metrics.get('cv_rmse_std', 0)
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance for a model"""
        if model_name is None:
            model_name = self.best_model_name or 'random_forest'
        
        if model_name not in self.feature_importance:
            return pd.DataFrame()
        
        importance_data = self.feature_importance[model_name]
        importance_df = pd.DataFrame(list(importance_data.items()), 
                                   columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, models_dir: str = None):
        """Save trained models and preprocessor"""
        if models_dir is None:
            models_dir = MODEL_CONFIG['models_dir']
        
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = models_dir / f"price_prediction_{model_name}.joblib"
            joblib.dump(model, model_path)
            logging.info(f"Saved {model_name} to {model_path}")
        
        # Save preprocessor
        preprocessor_path = models_dir / "price_prediction_preprocessor.joblib"
        self.preprocessor.save_preprocessors(preprocessor_path)
        
        # Save metadata
        metadata = {
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'best_model_name': self.best_model_name
        }
        metadata_path = models_dir / "price_prediction_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        logging.info("All price prediction models saved successfully")
    
    def load_models(self, models_dir: str = None):
        """Load trained models and preprocessor"""
        if models_dir is None:
            models_dir = MODEL_CONFIG['models_dir']
        
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            logging.warning(f"Models directory {models_dir} does not exist")
            return
        
        # Load models
        for model_name in self.models.keys():
            model_path = models_dir / f"price_prediction_{model_name}.joblib"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logging.info(f"Loaded {model_name} from {model_path}")
        
        # Load preprocessor
        preprocessor_path = models_dir / "price_prediction_preprocessor.joblib"
        if preprocessor_path.exists():
            self.preprocessor.load_preprocessors(preprocessor_path)
        
        # Load metadata
        metadata_path = models_dir / "price_prediction_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_performance = metadata.get('model_performance', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.best_model_name = metadata.get('best_model_name')
        
        logging.info("Price prediction models loaded successfully")
    
    def evaluate_on_new_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate models on new test data"""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                results[model_name] = metrics
                
                logging.info(f"{model_name} evaluation - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results