"""
Data generation utilities for creating sample e-commerce data
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import logging

# Try to import Faker, provide fallback if not available
try:
    from faker import Faker
except ImportError:
    # Fallback implementation if Faker is not available
    class Faker:
        def __init__(self):
            pass
        
        def seed(self, seed):
            pass
        
        def user_name(self):
            return f"user_{random.randint(1000, 9999)}"
        
        def email(self):
            domains = ['example.com', 'test.com', 'demo.com']
            return f"user{random.randint(100, 999)}@{random.choice(domains)}"
        
        def catch_phrase(self):
            phrases = ['Amazing Product', 'Premium Quality', 'Best Choice', 'Top Rated']
            return random.choice(phrases)
        
        def name(self):
            first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa']
            last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Moore']
            return f"{random.choice(first_names)} {random.choice(last_names)}"
        
        def date_time_between(self, start_date, end_date):
            # Simple random date generation
            days_ago = random.randint(1, 365)
            return datetime.now() - timedelta(days=days_ago)

from config.config import DATA_CONFIG
from data.database import DatabaseManager

class DataGenerator:
    """Generates realistic sample data for the e-commerce system"""
    
    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)
        random.seed(42)
        np.random.seed(42)
        
        self.categories = DATA_CONFIG['categories']
        self.brands = DATA_CONFIG['brands']
        self.price_ranges = DATA_CONFIG['price_ranges']
    
    def generate_products(self, num_products: int = 1000) -> List[Dict[str, Any]]:
        """Generate realistic product data"""
        products = []
        
        # Product name templates by category
        product_templates = {
            'Electronics': [
                '{brand} {model} {type}',
                '{brand} {model} {size}" {type}',
                '{brand} {type} {model}'
            ],
            'Clothing': [
                '{brand} {style} {item}',
                '{brand} {material} {item}',
                '{style} {item} by {brand}'
            ],
            'Home & Garden': [
                '{brand} {type} {item}',
                '{material} {item}',
                '{brand} {size} {item}'
            ],
            'Sports': [
                '{brand} {type} {item}',
                '{brand} {sport} {item}',
                '{type} {item} - {brand}'
            ],
            'Books': [
                '{title} by {author}',
                'The {adjective} {noun}',
                '{genre} {type}'
            ],
            'Beauty': [
                '{brand} {type} {product}',
                '{brand} {benefit} {product}',
                '{type} {product} - {brand}'
            ],
            'Automotive': [
                '{brand} {type} {part}',
                '{brand} {model} {part}',
                '{type} {part} for {brand}'
            ],
            'Toys': [
                '{brand} {type} {toy}',
                '{adjective} {toy}',
                '{toy} {type} Set'
            ],
            'Health': [
                '{brand} {type} {product}',
                '{benefit} {product}',
                '{brand} {form} {product}'
            ],
            'Food': [
                '{brand} {type} {food}',
                '{adjective} {food}',
                '{brand} {cuisine} {food}'
            ]
        }
        
        # Template variables
        template_vars = {
            'Electronics': {
                'type': ['Smartphone', 'Laptop', 'Tablet', 'TV', 'Camera', 'Headphones', 'Speaker', 'Monitor'],
                'model': ['Pro', 'Max', 'Ultra', 'Plus', 'Mini', 'Air', 'Standard'],
                'size': ['13', '15', '17', '21', '24', '27', '32', '55', '65']
            },
            'Clothing': {
                'item': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 'Hoodie', 'Pants', 'Shirt'],
                'style': ['Casual', 'Formal', 'Vintage', 'Modern', 'Classic', 'Trendy'],
                'material': ['Cotton', 'Denim', 'Leather', 'Silk', 'Wool', 'Polyester']
            },
            'Home & Garden': {
                'item': ['Chair', 'Table', 'Lamp', 'Vase', 'Plant', 'Tool', 'Decoration', 'Storage'],
                'type': ['Modern', 'Vintage', 'Rustic', 'Contemporary', 'Traditional'],
                'material': ['Wood', 'Metal', 'Plastic', 'Glass', 'Ceramic'],
                'size': ['Small', 'Medium', 'Large', 'XL']
            }
        }
        
        for i in range(num_products):
            category = random.choice(self.categories)
            brand = random.choice(self.brands + ['Generic', 'Store Brand'])
            
            # Generate product name
            if category in product_templates:
                template = random.choice(product_templates[category])
                vars_dict = template_vars.get(category, {})
                
                # Fill template variables
                name_parts = {}
                name_parts['brand'] = brand
                
                for var_type, options in vars_dict.items():
                    name_parts[var_type] = random.choice(options)
                
                # Add some generic options for missing variables
                generic_vars = {
                    'model': ['Model ' + str(random.randint(100, 999))],
                    'adjective': ['Premium', 'Deluxe', 'Professional', 'Advanced', 'Basic'],
                    'title': [self.fake.catch_phrase()],
                    'author': [self.fake.name()],
                    'noun': ['Guide', 'Manual', 'Story', 'Adventure'],
                    'genre': ['Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 'Biography'],
                    'benefit': ['Anti-Aging', 'Moisturizing', 'Brightening', 'Nourishing'],
                    'part': ['Filter', 'Belt', 'Battery', 'Oil', 'Brake Pad'],
                    'toy': ['Robot', 'Doll', 'Car', 'Puzzle', 'Game'],
                    'sport': ['Running', 'Basketball', 'Soccer', 'Tennis', 'Golf'],
                    'food': ['Snack', 'Beverage', 'Sauce', 'Cereal', 'Pasta'],
                    'cuisine': ['Italian', 'Asian', 'Mexican', 'American', 'Mediterranean'],
                    'form': ['Capsule', 'Liquid', 'Powder', 'Tablet', 'Cream']
                }
                
                for var in template.split():
                    var_clean = var.strip('{}')
                    if var_clean not in name_parts and var_clean in generic_vars:
                        name_parts[var_clean] = random.choice(generic_vars[var_clean])
                
                try:
                    name = template.format(**name_parts)
                except KeyError:
                    name = f"{brand} {category} Product {i+1}"
            else:
                name = f"{brand} {category} Product {i+1}"
            
            # Generate price based on category
            price_min, price_max = self.price_ranges.get(category, (10, 100))
            base_price = round(random.uniform(price_min, price_max), 2)
            
            # Adjust price based on brand (premium brands cost more)
            if brand in ['Apple', 'Sony', 'Nike', 'Adidas']:
                base_price *= random.uniform(1.2, 1.8)
            elif brand == 'Generic':
                base_price *= random.uniform(0.6, 0.9)
            
            # Generate features
            features = self._generate_product_features(category)
            
            # Generate description
            description = self._generate_description(name, category, features)
            
            # Generate rating and reviews
            rating = round(random.uniform(3.0, 5.0), 1)
            num_reviews = random.randint(10, 1000)
            
            product = {
                'name': name,
                'category': category,
                'brand': brand,
                'price': round(base_price, 2),
                'description': description,
                'features': features,
                'rating': rating,
                'num_reviews': num_reviews
            }
            
            products.append(product)
        
        return products
    
    def _generate_product_features(self, category: str) -> Dict[str, Any]:
        """Generate category-specific features"""
        features = {}
        
        feature_templates = {
            'Electronics': {
                'screen_size': lambda: f"{random.uniform(5, 65):.1f} inches",
                'storage': lambda: f"{random.choice([16, 32, 64, 128, 256, 512, 1024])}GB",
                'ram': lambda: f"{random.choice([4, 8, 16, 32])}GB",
                'color': lambda: random.choice(['Black', 'White', 'Silver', 'Gold', 'Blue', 'Red']),
                'wireless': lambda: random.choice(['WiFi', 'Bluetooth', 'Both', 'None']),
                'warranty': lambda: f"{random.randint(1, 3)} year(s)"
            },
            'Clothing': {
                'size': lambda: random.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL']),
                'color': lambda: random.choice(['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Navy']),
                'material': lambda: random.choice(['Cotton', 'Polyester', 'Wool', 'Silk', 'Denim', 'Leather']),
                'fit': lambda: random.choice(['Regular', 'Slim', 'Loose', 'Tight']),
                'care': lambda: random.choice(['Machine Wash', 'Hand Wash', 'Dry Clean Only'])
            },
            'Home & Garden': {
                'dimensions': lambda: f"{random.randint(10, 100)}x{random.randint(10, 100)}x{random.randint(10, 100)}cm",
                'material': lambda: random.choice(['Wood', 'Metal', 'Plastic', 'Glass', 'Fabric']),
                'color': lambda: random.choice(['Brown', 'Black', 'White', 'Natural', 'Gray']),
                'assembly_required': lambda: random.choice(['Yes', 'No']),
                'room': lambda: random.choice(['Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Garden'])
            }
        }
        
        if category in feature_templates:
            template = feature_templates[category]
            # Select 3-5 random features for each product
            selected_features = random.sample(list(template.keys()), min(len(template), random.randint(3, 5)))
            
            for feature_name in selected_features:
                features[feature_name] = template[feature_name]()
        
        # Add some common features
        features['weight'] = f"{random.uniform(0.1, 50):.1f} kg"
        features['in_stock'] = random.choice([True, False])
        features['free_shipping'] = random.choice([True, False])
        
        return features
    
    def _generate_description(self, name: str, category: str, features: Dict[str, Any]) -> str:
        """Generate product description"""
        descriptions = [
            f"Discover the amazing {name}, perfect for all your {category.lower()} needs.",
            f"Premium quality {name} designed with the latest technology.",
            f"Experience excellence with this high-quality {name}.",
            f"The {name} combines style, functionality, and durability.",
            f"Upgrade your lifestyle with the innovative {name}."
        ]
        
        base_desc = random.choice(descriptions)
        
        # Add feature highlights
        feature_highlights = []
        for key, value in list(features.items())[:3]:  # Highlight first 3 features
            feature_highlights.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if feature_highlights:
            base_desc += " Key features include: " + ", ".join(feature_highlights) + "."
        
        return base_desc
    
    def generate_users(self, num_users: int = 500) -> List[Dict[str, Any]]:
        """Generate realistic user data"""
        users = []
        
        for i in range(num_users):
            user = {
                'username': self.fake.user_name() + str(random.randint(1, 999)),
                'email': self.fake.email(),
                'age': random.randint(18, 80),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'preferences': {
                    'preferred_categories': random.sample(self.categories, random.randint(1, 4)),
                    'price_sensitivity': random.choice(['Low', 'Medium', 'High']),
                    'brand_loyalty': random.choice(['Low', 'Medium', 'High']),
                    'shopping_frequency': random.choice(['Weekly', 'Monthly', 'Occasionally'])
                }
            }
            users.append(user)
        
        return users
    
    def generate_interactions(self, users: List[Dict], products: List[Dict], 
                            num_interactions: int = 5000) -> List[Dict[str, Any]]:
        """Generate user-product interactions"""
        interactions = []
        
        interaction_types = ['view', 'cart', 'purchase', 'rating']
        type_weights = [0.5, 0.2, 0.2, 0.1]  # Views are most common
        
        for i in range(num_interactions):
            user = random.choice(users)
            product = random.choice(products)
            interaction_type = np.random.choice(interaction_types, p=type_weights)
            
            interaction = {
                'user_id': user.get('id', random.randint(1, len(users))),
                'product_id': product.get('id', random.randint(1, len(products))),
                'interaction_type': interaction_type,
                'timestamp': self.fake.date_time_between(start_date='-1y', end_date='now')
            }
            
            # Add specific data based on interaction type
            if interaction_type == 'purchase':
                interaction['quantity'] = random.randint(1, 5)
                interaction['price_paid'] = product['price'] * random.uniform(0.8, 1.2)  # Some price variation
            elif interaction_type == 'rating':
                interaction['rating'] = random.randint(1, 5)
            elif interaction_type == 'cart':
                interaction['quantity'] = random.randint(1, 3)
            
            interactions.append(interaction)
        
        return interactions
    
    def populate_database(self, db_manager: DatabaseManager, 
                         num_products: int = 1000, 
                         num_users: int = 500, 
                         num_interactions: int = 5000):
        """Populate database with sample data"""
        
        logging.info("Generating sample products...")
        products = self.generate_products(num_products)
        
        logging.info("Generating sample users...")
        users = self.generate_users(num_users)
        
        logging.info("Adding products to database...")
        product_ids = []
        for product in products:
            product_id = db_manager.add_product(product)
            product['id'] = product_id
            product_ids.append(product_id)
        
        logging.info("Adding users to database...")
        user_ids = []
        for user in users:
            user_id = db_manager.add_user(user)
            user['id'] = user_id
            user_ids.append(user_id)
        
        logging.info("Generating and adding interactions...")
        interactions = self.generate_interactions(users, products, num_interactions)
        
        for interaction in interactions:
            db_manager.add_interaction(interaction)
        
        # Add categories
        logging.info("Adding categories...")
        for category in self.categories:
            with db_manager.get_connection() as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO categories (name, description)
                    VALUES (?, ?)
                ''', (category, f"Products in the {category} category"))
        
        logging.info(f"Database populated with {len(products)} products, {len(users)} users, and {len(interactions)} interactions.")
        
        return {
            'products': len(products),
            'users': len(users),
            'interactions': len(interactions)
        }

def create_sample_data():
    """Convenience function to create sample data"""
    db_manager = DatabaseManager()
    generator = DataGenerator()
    
    # Clear existing data
    db_manager.clear_all_data()
    
    # Generate new sample data
    stats = generator.populate_database(
        db_manager,
        num_products=1000,
        num_users=500,
        num_interactions=5000
    )
    
    return stats

if __name__ == "__main__":
    # Generate sample data when run directly
    logging.basicConfig(level=logging.INFO)
    stats = create_sample_data()
    print("Sample data generated successfully!")
    print(f"Created: {stats}")