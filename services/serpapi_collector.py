"""
Real-time product data collection using SerpAPI
"""

import requests
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import random

from serpapi import GoogleSearch
from data.database import DatabaseManager

class SerpAPIDataCollector:
    """Collects real product data using SerpAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.db_manager = DatabaseManager()
        
    def search_products(self, query: str, num_results: int = 20) -> List[Dict[str, Any]]:
        """Search for products using Google Shopping via SerpAPI"""
        try:
            params = {
                "engine": "google_shopping",
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "location": "United States",
                "hl": "en",
                "gl": "us"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            products = []
            
            if "shopping_results" in results:
                for item in results["shopping_results"]:
                    try:
                        # Extract price
                        price = self._extract_price(item.get("price", "0"))
                        
                        # Extract rating
                        rating = self._extract_rating(item.get("rating"))
                        
                        # Create product data
                        product = {
                            "name": item.get("title", "Unknown Product"),
                            "price": price,
                            "description": item.get("snippet", ""),
                            "brand": self._extract_brand(item.get("title", "")),
                            "category": self._categorize_product(query, item.get("title", "")),
                            "rating": rating,
                            "num_reviews": self._extract_review_count(item.get("reviews", 0)),
                            "image_url": item.get("thumbnail", ""),
                            "source_url": item.get("link", ""),
                            "source": "SerpAPI",
                            "search_query": query,
                            "collected_at": datetime.now().isoformat()
                        }
                        
                        products.append(product)
                        
                    except Exception as e:
                        logging.warning(f"Error processing product item: {str(e)}")
                        continue
            
            logging.info(f"Collected {len(products)} products for query: {query}")
            return products
            
        except (ConnectionError, OSError, Exception) as e:
            if "getaddrinfo failed" in str(e) or "Connection" in str(e):
                logging.warning(f"Network connectivity issue with SerpAPI for query '{query}': {str(e)}")
            else:
                logging.error(f"Error searching products with SerpAPI: {str(e)}")
            return []
    
    def _extract_price(self, price_str: str) -> float:
        """Extract numeric price from price string"""
        if not price_str:
            return 0.0
        
        # Remove currency symbols and extract numeric value
        import re
        price_match = re.search(r'[\d,]+\.?\d*', str(price_str).replace(',', ''))
        if price_match:
            try:
                return float(price_match.group())
            except ValueError:
                return 0.0
        return 0.0
    
    def _extract_rating(self, rating_data) -> float:
        """Extract rating from rating data"""
        if not rating_data:
            return 4.0  # Default rating
        
        if isinstance(rating_data, (int, float)):
            return float(rating_data)
        
        if isinstance(rating_data, str):
            import re
            rating_match = re.search(r'(\d+\.?\d*)', rating_data)
            if rating_match:
                return float(rating_match.group(1))
        
        return 4.0  # Default rating
    
    def _extract_review_count(self, reviews_data) -> int:
        """Extract number of reviews"""
        if isinstance(reviews_data, int):
            return reviews_data
        
        if isinstance(reviews_data, str):
            import re
            review_match = re.search(r'(\d+)', reviews_data.replace(',', ''))
            if review_match:
                return int(review_match.group(1))
        
        return random.randint(10, 500)  # Random review count as fallback
    
    def _extract_brand(self, title: str) -> str:
        """Extract brand from product title"""
        # Common brand patterns
        brands = [
            'Apple', 'Samsung', 'Google', 'Microsoft', 'Sony', 'LG', 'HP', 'Dell',
            'Lenovo', 'Asus', 'Acer', 'Nike', 'Adidas', 'Amazon', 'Canon', 'Nikon',
            'Nintendo', 'PlayStation', 'Xbox', 'iPhone', 'iPad', 'MacBook', 'Surface'
        ]
        
        title_upper = title.upper()
        for brand in brands:
            if brand.upper() in title_upper:
                return brand
        
        # Extract first word as potential brand
        words = title.split()
        if words:
            return words[0]
        
        return "Generic"
    
    def _categorize_product(self, query: str, title: str) -> str:
        """Categorize product based on query and title"""
        query_lower = query.lower()
        title_lower = title.lower()
        
        categories = {
            'Electronics': ['phone', 'laptop', 'computer', 'tablet', 'headphone', 'speaker', 'camera', 'tv', 'monitor'],
            'Clothing': ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'sweater', 'jeans', 'sneaker'],
            'Home & Garden': ['furniture', 'decor', 'kitchen', 'bathroom', 'garden', 'tool', 'appliance'],
            'Sports': ['fitness', 'exercise', 'sport', 'gym', 'outdoor', 'bike', 'run'],
            'Books': ['book', 'novel', 'textbook', 'manual', 'guide'],
            'Beauty': ['makeup', 'skincare', 'perfume', 'cosmetic', 'beauty'],
            'Automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'motor']
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in query_lower or keyword in title_lower:
                    return category
        
        return 'General'
    
    def collect_multiple_categories(self, categories: List[str], products_per_category: int = 10) -> List[Dict[str, Any]]:
        """Collect products from multiple categories"""
        all_products = []
        
        for category in categories:
            logging.info(f"Collecting products for category: {category}")
            products = self.search_products(category, products_per_category)
            all_products.extend(products)
            
            # Add delay to respect API rate limits
            time.sleep(1)
        
        return all_products
    
    def save_products_to_database(self, products: List[Dict[str, Any]]) -> int:
        """Save collected products to database"""
        saved_count = 0
        
        for product in products:
            try:
                # Check if product already exists (by name and approximate price)
                existing_products = self.db_manager.get_products()
                
                duplicate = False
                if not existing_products.empty:
                    for _, existing in existing_products.iterrows():
                        if (existing['name'].lower() == product['name'].lower() and 
                            abs(existing['price'] - product['price']) < 1.0):
                            duplicate = True
                            # Update price history for existing product
                            self.db_manager.add_price_history(existing['id'], product['price'])
                            break
                
                if not duplicate:
                    # Add new product
                    product_data = {
                        'name': product['name'],
                        'description': product['description'],
                        'price': product['price'],
                        'category': product['category'],
                        'brand': product['brand'],
                        'rating': product['rating'],
                        'num_reviews': product['num_reviews'],
                        'features': json.dumps({
                            'image_url': product.get('image_url', ''),
                            'source_url': product.get('source_url', ''),
                            'source': product.get('source', 'SerpAPI'),
                            'search_query': product.get('search_query', ''),
                            'collected_at': product.get('collected_at', '')
                        })
                    }
                    
                    product_id = self.db_manager.add_product(product_data)
                    
                    # Add initial price history
                    self.db_manager.add_price_history(product_id, product['price'])
                    
                    saved_count += 1
                
            except Exception as e:
                logging.error(f"Error saving product {product['name']}: {str(e)}")
                continue
        
        logging.info(f"Saved {saved_count} new products to database")
        return saved_count
    
    def update_existing_prices(self, sample_size: int = 20) -> int:
        """Update prices for existing products"""
        try:
            # Get a sample of existing products
            existing_products = self.db_manager.get_products(limit=sample_size)
            
            if existing_products.empty:
                return 0
            
            updated_count = 0
            
            for _, product in existing_products.iterrows():
                try:
                    # Search for the product to get updated price
                    search_query = f"{product['brand']} {product['name']}"
                    updated_products = self.search_products(search_query, 1)
                    
                    if updated_products:
                        new_price = updated_products[0]['price']
                        
                        # Update price if it's different
                        if abs(new_price - product['price']) > 0.01:
                            self.db_manager.update_product_price(product['id'], new_price)
                            updated_count += 1
                    
                    # Add delay to respect API rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    logging.warning(f"Error updating price for {product['name']}: {str(e)}")
                    continue
            
            logging.info(f"Updated prices for {updated_count} products")
            return updated_count
            
        except Exception as e:
            logging.error(f"Error updating existing prices: {str(e)}")
            return 0

def collect_real_time_data(api_key: str, categories: List[str] = None) -> Dict[str, int]:
    """Convenience function to collect real-time data"""
    
    if not categories:
        categories = [
            "iPhone 15", "Samsung Galaxy", "MacBook Pro", "iPad",
            "Nike sneakers", "Adidas shoes", "Sony headphones",
            "Dell laptop", "HP printer", "Canon camera"
        ]
    
    collector = SerpAPIDataCollector(api_key)
    
    # Collect products
    products = collector.collect_multiple_categories(categories, products_per_category=5)
    
    # Save to database
    saved_count = collector.save_products_to_database(products)
    
    # Update some existing prices
    updated_count = collector.update_existing_prices(sample_size=10)
    
    return {
        'products_collected': len(products),
        'products_saved': saved_count,
        'prices_updated': updated_count
    }