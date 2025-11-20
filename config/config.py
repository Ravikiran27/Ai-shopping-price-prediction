"""
Configuration settings for the E-Commerce Price Prediction System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database configuration
DATABASE_CONFIG = {
    'database_path': BASE_DIR / 'data' / 'ecommerce.db',
    'backup_path': BASE_DIR / 'data' / 'backups'
}

# Model configuration
MODEL_CONFIG = {
    'models_dir': BASE_DIR / 'models' / 'saved_models',
    'price_prediction': {
        'algorithms': ['random_forest', 'gradient_boosting', 'linear_regression'],
        'default_algorithm': 'random_forest',
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5
    },
    'recommendation': {
        'collaborative_filtering': {
            'n_factors': 50,
            'n_epochs': 100,
            'lr_all': 0.005,
            'reg_all': 0.02
        },
        'content_based': {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'stop_words': 'english'
        },
        'hybrid': {
            'collaborative_weight': 0.6,
            'content_weight': 0.4
        }
    }
}

# Data configuration
DATA_CONFIG = {
    'sample_size': 10000,
    'categories': [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books',
        'Beauty', 'Automotive', 'Toys', 'Health', 'Food'
    ],
    'brands': [
        'Apple', 'Samsung', 'Sony', 'Nike', 'Adidas', 'Amazon',
        'Google', 'Microsoft', 'Dell', 'HP', 'Canon', 'Nikon',
        'LG', 'Panasonic', 'Philips', 'Bosch', 'Generic'
    ],
    'price_ranges': {
        'Electronics': (50, 2000),
        'Clothing': (20, 500),
        'Home & Garden': (30, 1000),
        'Sports': (25, 800),
        'Books': (10, 100),
        'Beauty': (15, 300),
        'Automotive': (100, 5000),
        'Toys': (10, 200),
        'Health': (20, 400),
        'Food': (5, 100)
    }
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'AI-Powered E-Commerce Analytics',
    'page_icon': '🛒',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#FF6B6B',
        'backgroundColor': '#FFFFFF',
        'secondaryBackgroundColor': '#F0F0F0',
        'textColor': '#262730'
    }
}

# Visualization configuration
VIZ_CONFIG = {
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
    'plot_style': 'plotly_white',
    'figure_size': (12, 8),
    'dpi': 100
}

# API configuration (for future extensions)
API_CONFIG = {
    'rate_limit': 100,  # requests per minute
    'timeout': 30,      # seconds
    'retry_attempts': 3,
    'serpapi_key': '9a6457f442af64cd66e37f0fbafb116801060af75662d860bb8d68700ec01320'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': BASE_DIR / 'logs' / 'app.log'
}

# Security configuration
SECURITY_CONFIG = {
    'session_timeout': 3600,  # seconds
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_file_types': ['.csv', '.xlsx', '.json']
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'cache_ttl': 3600,  # seconds
    'max_cache_size': 100,  # MB
    'batch_size': 1000,
    'max_workers': 4
}