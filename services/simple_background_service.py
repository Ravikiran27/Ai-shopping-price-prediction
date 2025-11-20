"""
Simple background service for automatic data collection and model training
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import random

from data.database import DatabaseManager
from data.data_generator import DataGenerator
from models.price_predictor import PricePredictionModel
from models.time_series_predictor import TimeSeriesPricePredictor
from services.serpapi_collector import SerpAPIDataCollector
from config.config import API_CONFIG

class SimpleBackgroundService:
    """Simple background service for automatic operations"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.price_model = PricePredictionModel()
        self.ts_model = TimeSeriesPricePredictor()
        self.serpapi_collector = SerpAPIDataCollector(API_CONFIG['serpapi_key'])
        self.running = False
        self.thread = None
        self.last_data_generation = None
        self.last_model_training = None
        self.last_real_data_collection = None
        
    def start(self):
        """Start the background service"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_background_tasks, daemon=True)
            self.thread.start()
            logging.info("Background service started")
    
    def stop(self):
        """Stop the background service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logging.info("Background service stopped")
    
    def _run_background_tasks(self):
        """Main background task loop"""
        while self.running:
            try:
                # Check if we need to initialize data
                self._ensure_sample_data()
                
                # Collect real-time data periodically (every 30 minutes)
                self._collect_real_time_data()
                
                # Generate new price history periodically (every 5 minutes for demo)
                self._update_price_history()
                
                # Train models periodically (every 15 minutes for demo)
                self._train_models_if_needed()
                
                # Sleep for 60 seconds before next check
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Error in background service: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _ensure_sample_data(self):
        """Ensure we have sample data in the database"""
        try:
            stats = self.db_manager.get_database_stats()
            
            # If we have very little data, generate some
            if stats.get('products', 0) < 20:
                logging.info("Generating initial sample data...")
                # Use the data generator to create sample data
                generator = DataGenerator()
                generator.populate_database(
                    self.db_manager,
                    num_products=50,
                    num_users=30,
                    num_interactions=200
                )
                self.last_data_generation = datetime.now()
                logging.info("Initial sample data generated")
                
        except Exception as e:
            logging.error(f"Error ensuring sample data: {str(e)}")
    
    def _collect_real_time_data(self):
        """Collect real-time product data using SerpAPI"""
        try:
            # Only collect real data if it's been more than 30 minutes since last collection
            if (self.last_real_data_collection and 
                datetime.now() - self.last_real_data_collection < timedelta(minutes=30)):
                return
            
            logging.info("Starting real-time data collection...")
            
            # Define popular product categories to search
            search_queries = [
                "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3",
                "iPad Pro", "AirPods Pro", "Sony WH-1000XM5",
                "Nike Air Max", "Adidas Ultraboost", "Dell XPS 13",
                "HP Pavilion", "Canon EOS R5", "Sony A7 IV"
            ]
            
            # Collect a few products per cycle to respect API limits
            selected_queries = random.sample(search_queries, min(3, len(search_queries)))
            
            total_collected = 0
            total_updated = 0
            
            for query in selected_queries:
                try:
                    # Collect new products
                    products = self.serpapi_collector.search_products(query, 3)
                    saved_count = self.serpapi_collector.save_products_to_database(products)
                    total_collected += saved_count
                    
                    # Small delay between queries
                    time.sleep(2)
                    
                except Exception as e:
                    logging.warning(f"Error collecting data for query '{query}': {str(e)}")
                    continue
            
            # Update prices for some existing products
            updated_count = self.serpapi_collector.update_existing_prices(sample_size=5)
            total_updated += updated_count
            
            self.last_real_data_collection = datetime.now()
            logging.info(f"Real-time data collection completed. Collected: {total_collected}, Updated: {total_updated}")
            
        except Exception as e:
            logging.warning(f"Real-time data collection skipped due to network issues: {str(e)}")
            # Don't treat network issues as critical errors
    
    def _update_price_history(self):
        """Update price history for products"""
        try:
            # Only update if it's been more than 5 minutes since last update
            if (self.last_data_generation and 
                datetime.now() - self.last_data_generation < timedelta(minutes=5)):
                return
            
            # Get some products to update prices
            products_df = self.db_manager.get_products(limit=10)
            
            for _, product in products_df.iterrows():
                # Add some price variation (±2%)
                current_price = product['price']
                price_change = random.uniform(-0.02, 0.02)
                new_price = current_price * (1 + price_change)
                
                # Ensure price stays reasonable
                new_price = max(current_price * 0.8, min(current_price * 1.2, new_price))
                
                # Update price in database (this will automatically add to price_history)
                self.db_manager.update_product_price(product['id'], new_price)
            
            self.last_data_generation = datetime.now()
            logging.info(f"Updated price history for {len(products_df)} products")
            
        except Exception as e:
            logging.error(f"Error updating price history: {str(e)}")
    
    def _train_models_if_needed(self):
        """Train models if needed"""
        try:
            # Only train if it's been more than 10 minutes since last training
            if (self.last_model_training and 
                datetime.now() - self.last_model_training < timedelta(minutes=10)):
                return
            
            # Check if we have enough data
            stats = self.db_manager.get_database_stats()
            
            if stats.get('products', 0) >= 50:
                # Train traditional models
                products_df = self.db_manager.get_products()
                X, y = self.price_model.prepare_data(products_df)
                self.price_model.train_models(X, y)
                self.price_model.save_models()
                logging.info("Traditional models trained")
            
            # Train time-series model if we have price history (reduced requirement)
            price_history = self.db_manager.get_price_history()
            if len(price_history) >= 20:  # Reduced from 50 to 20
                try:
                    products_df = self.db_manager.get_products()
                    self.ts_model.train(price_history, products_df)
                    self.ts_model.save_model("models/saved_models/time_series_model.joblib")
                    logging.info("Time-series model trained")
                except Exception as e:
                    logging.warning(f"Time-series model training skipped: {str(e)}")
            
            self.last_model_training = datetime.now()
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
    
    def get_status(self) -> dict:
        """Get background service status"""
        try:
            stats = self.db_manager.get_database_stats()
            price_history = self.db_manager.get_price_history()
            price_history_count = len(price_history) if not price_history.empty else 0
            
            return {
                'running': self.running,
                'products': stats.get('products', 0),
                'users': stats.get('users', 0),
                'interactions': stats.get('interactions', 0),
                'price_history_points': price_history_count,
                'last_data_update': self.last_data_generation.strftime('%Y-%m-%d %H:%M:%S') if self.last_data_generation else 'Never',
                'last_model_training': self.last_model_training.strftime('%Y-%m-%d %H:%M:%S') if self.last_model_training else 'Never',
                'last_real_data_collection': self.last_real_data_collection.strftime('%Y-%m-%d %H:%M:%S') if self.last_real_data_collection else 'Never'
            }
        except Exception as e:
            logging.error(f"Error getting status: {str(e)}")
            return {
                'running': self.running,
                'products': 0,
                'users': 0,
                'interactions': 0,
                'price_history_points': 0,
                'last_data_update': 'Error',
                'last_model_training': 'Error',
                'last_real_data_collection': 'Error'
            }

# Global background service instance
_background_service = None

def get_background_service() -> SimpleBackgroundService:
    """Get the global background service instance"""
    global _background_service
    if _background_service is None:
        _background_service = SimpleBackgroundService()
    return _background_service

def start_background_service():
    """Start the background service"""
    service = get_background_service()
    service.start()
    return service

def stop_background_service():
    """Stop the background service"""
    global _background_service
    if _background_service:
        _background_service.stop()
        _background_service = None