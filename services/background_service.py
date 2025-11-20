"""
Background services for automatic data collection, model training, and predictions
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio
import schedule

from data.database import DatabaseManager
from data.data_generator import create_sample_data
from models.price_predictor import PricePredictionModel
from models.time_series_predictor import TimeSeriesPricePredictor

class BackgroundService:
    """Handles all background operations for the e-commerce system"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.price_model = PricePredictionModel()
        self.ts_model = TimeSeriesPricePredictor()
        self.is_running = False
        self.last_training = None
        self.last_data_update = None
        
    def start_background_services(self):
        """Start all background services"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Schedule background tasks
        schedule.every(1).hours.do(self.update_price_data)
        schedule.every(6).hours.do(self.train_models)
        schedule.every(30).minutes.do(self.generate_predictions)
        schedule.every(24).hours.do(self.cleanup_old_data)
        
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Initial setup
        self._initial_setup()
        
        logging.info("Background services started successfully")
    
    def stop_background_services(self):
        """Stop all background services"""
        self.is_running = False
        schedule.clear()
        logging.info("Background services stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logging.error(f"Error in scheduler: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _initial_setup(self):
        """Perform initial setup when app starts"""
        try:
            # Ensure database is initialized
            self.db_manager.init_database()
            
            # Check if we have sample data, if not create it
            products_df = self.db_manager.get_products()
            if len(products_df) < 50:
                logging.info("Creating initial sample data...")
                create_sample_data(self.db_manager, num_products=200, num_users=100, num_interactions=1000)
                self.generate_initial_price_history()
            
            # Check if models are trained, if not train them
            try:
                self.price_model.load_models()
                self.ts_model.load_model("models/saved_models/time_series_model.joblib")
            except:
                logging.info("Training initial models...")
                self.train_models()
            
            logging.info("Initial setup completed")
            
        except Exception as e:
            logging.error(f"Error in initial setup: {str(e)}")
    
    def update_price_data(self):
        """Update price data for products (simulate real-time price changes)"""
        try:
            products_df = self.db_manager.get_products()
            
            # Simulate price updates for 5-10% of products
            import random
            num_updates = max(1, len(products_df) // 10)
            products_to_update = products_df.sample(n=min(num_updates, len(products_df)))
            
            for _, product in products_to_update.iterrows():
                # Simulate realistic price changes (-3% to +3%)
                change_factor = random.uniform(0.97, 1.03)
                new_price = product['price'] * change_factor
                
                # Add some constraints to keep prices reasonable
                if new_price < product['price'] * 0.5:  # Don't go below 50% of original
                    new_price = product['price'] * 0.5
                elif new_price > product['price'] * 1.5:  # Don't go above 150% of original
                    new_price = product['price'] * 1.5
                
                self.db_manager.update_product_price(product['id'], new_price)
            
            self.last_data_update = datetime.now()
            logging.info(f"Updated prices for {len(products_to_update)} products")
            
        except Exception as e:
            logging.error(f"Error updating price data: {str(e)}")
    
    def train_models(self):
        """Train both traditional ML and time-series models"""
        try:
            # Train traditional ML models
            products_df = self.db_manager.get_products()
            if len(products_df) >= 50:
                X, y = self.price_model.prepare_data(products_df)
                self.price_model.train_models(X, y)
                self.price_model.save_models()
                logging.info("Traditional ML models trained and saved")
            
            # Train time-series model
            price_history = self.db_manager.get_price_history()
            if len(price_history) >= 50:
                self.ts_model.train(price_history, products_df)
                self.ts_model.save_model("models/saved_models/time_series_model.joblib")
                logging.info("Time-series model trained and saved")
            
            self.last_training = datetime.now()
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
    
    def generate_predictions(self):
        """Generate predictions for trending products"""
        try:
            # Get trending products (most viewed/purchased recently)
            interactions_df = self.db_manager.get_interactions(limit=1000)
            if not interactions_df.empty:
                trending_products = interactions_df.groupby('product_id').size().head(10).index.tolist()
                
                # Generate predictions for trending products
                for product_id in trending_products:
                    try:
                        price_history = self.db_manager.get_price_history(product_id)
                        if len(price_history) >= 10:  # Minimum history needed
                            products_df = self.db_manager.get_products()
                            product_info = products_df[products_df['id'] == product_id].iloc[0].to_dict()
                            
                            # Generate 7-day forecast
                            predictions = self.ts_model.predict_future_prices(
                                product_id, 7, price_history, product_info
                            )
                            # Store predictions if needed (could add predictions table)
                            
                    except Exception as e:
                        logging.warning(f"Failed to generate prediction for product {product_id}: {str(e)}")
            
            logging.info("Generated predictions for trending products")
            
        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
    
    def cleanup_old_data(self):
        """Clean up old data to maintain performance"""
        try:
            # Remove price history older than 1 year
            cutoff_date = datetime.now() - timedelta(days=365)
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute('''
                    DELETE FROM price_history 
                    WHERE timestamp < ?
                ''', (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logging.info(f"Cleaned up {deleted_count} old price history records")
            
        except Exception as e:
            logging.error(f"Error cleaning up old data: {str(e)}")
    
    def generate_initial_price_history(self, days_back: int = 90):
        """Generate initial price history for all products"""
        try:
            products_df = self.db_manager.get_products()
            import random
            
            for _, product in products_df.iterrows():
                base_price = product['price']
                current_price = base_price
                
                # Generate realistic price history
                for i in range(days_back):
                    # More volatility for electronics, less for books
                    if product['category'].lower() in ['electronics', 'computers']:
                        volatility = 0.02  # 2% daily volatility
                    elif product['category'].lower() in ['books', 'home']:
                        volatility = 0.005  # 0.5% daily volatility
                    else:
                        volatility = 0.01  # 1% daily volatility
                    
                    # Random walk with mean reversion
                    change = random.gauss(0, volatility)
                    # Add mean reversion (pull back to original price)
                    mean_reversion = (base_price - current_price) * 0.001
                    change += mean_reversion
                    
                    current_price *= (1 + change)
                    
                    # Keep price within reasonable bounds
                    current_price = max(base_price * 0.3, min(base_price * 2.5, current_price))
                    
                    timestamp = (datetime.now() - timedelta(days=days_back-i)).strftime('%Y-%m-%d %H:%M:%S')
                    self.db_manager.add_price_history(product['id'], current_price, timestamp)
            
            logging.info(f"Generated {days_back} days of price history for {len(products_df)} products")
            
        except Exception as e:
            logging.error(f"Error generating initial price history: {str(e)}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of background services"""
        return {
            'is_running': self.is_running,
            'last_training': self.last_training.strftime('%Y-%m-%d %H:%M:%S') if self.last_training else 'Never',
            'last_data_update': self.last_data_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_data_update else 'Never',
            'database_stats': self.db_manager.get_database_stats()
        }

# Global instance
background_service = BackgroundService()