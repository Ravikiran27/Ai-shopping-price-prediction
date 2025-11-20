"""
Helper functions and utilities for the e-commerce system
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from pathlib import Path

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency with proper symbol and formatting"""
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
    elif currency == 'GBP':
        return f"£{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_number(number: int) -> str:
    """Format large numbers with K, M, B suffixes"""
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with proper symbol"""
    return f"{value:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-\.,!?]', '', text)
    
    return text

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers from text"""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(num) for num in numbers]

def calculate_age_group(age: int) -> str:
    """Calculate age group from age"""
    if age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

def get_season(date: datetime) -> str:
    """Get season from date"""
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate business days between two dates"""
    return pd.bdate_range(start_date, end_date).shape[0]

def generate_color_palette(n_colors: int, palette_type: str = 'default') -> List[str]:
    """Generate a color palette with n colors"""
    palettes = {
        'default': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
        'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#F39C12'],
        'pastel': ['#FFD93D', '#6BCF7F', '#4D96FF', '#FF6B9D', '#C44536', '#F8B500', '#5D737E'],
        'dark': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7', '#D5DBDB', '#F8F9FA']
    }
    
    base_colors = palettes.get(palette_type, palettes['default'])
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    else:
        # Generate additional colors by modifying existing ones
        colors = base_colors.copy()
        while len(colors) < n_colors:
            # Add variations of existing colors
            for base_color in base_colors:
                if len(colors) >= n_colors:
                    break
                # Create a lighter/darker variation
                colors.append(lighten_color(base_color, 0.2))
        
        return colors[:n_colors]

def lighten_color(hex_color: str, factor: float) -> str:
    """Lighten a hex color by a factor (0-1)"""
    try:
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Lighten each component
        lightened_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
        
        # Convert back to hex
        return f"#{lightened_rgb[0]:02x}{lightened_rgb[1]:02x}{lightened_rgb[2]:02x}"
    
    except:
        return hex_color

def create_download_link(data: Any, filename: str, mime_type: str = 'text/csv') -> str:
    """Create a download link for data"""
    import base64
    
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    elif isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
    else:
        b64 = base64.b64encode(str(data).encode()).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

@st.cache_data
def load_and_cache_data(data_loader_func, *args, **kwargs):
    """Cache data loading function"""
    return data_loader_func(*args, **kwargs)

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions for analytics"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details or {},
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    logging.info(f"User action: {json.dumps(log_entry)}")

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digits
    digits_only = re.sub(r'\D', '', phone)
    return len(digits_only) >= 10

def calculate_discount_percentage(original_price: float, discounted_price: float) -> float:
    """Calculate discount percentage"""
    if original_price <= 0:
        return 0.0
    return ((original_price - discounted_price) / original_price) * 100

def get_price_category(price: float, price_ranges: Dict[str, Tuple[float, float]]) -> str:
    """Categorize price based on ranges"""
    for category, (min_price, max_price) in price_ranges.items():
        if min_price <= price <= max_price:
            return category
    return "Other"

def calculate_similarity_score(item1: Dict[str, Any], item2: Dict[str, Any], 
                             weights: Dict[str, float] = None) -> float:
    """Calculate similarity score between two items"""
    if weights is None:
        weights = {'category': 0.3, 'brand': 0.2, 'price': 0.3, 'rating': 0.2}
    
    score = 0.0
    total_weight = 0.0
    
    for feature, weight in weights.items():
        if feature in item1 and feature in item2:
            if feature in ['category', 'brand']:
                # Categorical similarity
                score += weight if item1[feature] == item2[feature] else 0
            elif feature in ['price', 'rating']:
                # Numerical similarity (inverse of normalized difference)
                val1, val2 = item1[feature], item2[feature]
                if feature == 'price':
                    max_val = max(val1, val2, 1)  # Avoid division by zero
                    similarity = 1 - abs(val1 - val2) / max_val
                else:  # rating
                    similarity = 1 - abs(val1 - val2) / 5  # Assuming 5-point scale
                score += weight * max(0, similarity)
            
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Chunk a list into smaller lists of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_file_size(file_path: str) -> str:
    """Get human readable file size"""
    try:
        size_bytes = Path(file_path).stat().st_size
        
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(np.floor(np.log2(size_bytes) / 10))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    except:
        return "Unknown"

def is_business_hour(dt: datetime, start_hour: int = 9, end_hour: int = 17) -> bool:
    """Check if datetime is within business hours"""
    return start_hour <= dt.hour < end_hour and dt.weekday() < 5

def calculate_growth_rate(current_value: float, previous_value: float) -> float:
    """Calculate growth rate percentage"""
    if previous_value == 0:
        return 0.0 if current_value == 0 else 100.0
    
    return ((current_value - previous_value) / previous_value) * 100

def get_trend_direction(values: List[float], window: int = 3) -> str:
    """Determine trend direction from a series of values"""
    if len(values) < window:
        return "Unknown"
    
    recent_values = values[-window:]
    
    if len(set(recent_values)) == 1:
        return "Stable"
    
    # Calculate simple trend
    increases = sum(1 for i in range(1, len(recent_values)) 
                   if recent_values[i] > recent_values[i-1])
    decreases = sum(1 for i in range(1, len(recent_values)) 
                   if recent_values[i] < recent_values[i-1])
    
    if increases > decreases:
        return "Increasing"
    elif decreases > increases:
        return "Decreasing"
    else:
        return "Volatile"

def create_summary_stats(data: pd.Series) -> Dict[str, Any]:
    """Create summary statistics for a pandas Series"""
    if data.empty:
        return {}
    
    stats = {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
    }
    
    # Add mode for categorical data
    if data.dtype == 'object':
        stats['mode'] = data.mode().iloc[0] if not data.mode().empty else None
        stats['unique_count'] = data.nunique()
    
    return stats

class Timer:
    """Simple timer context manager"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        logging.info(f"{self.description} completed in {duration.total_seconds():.2f} seconds")

def memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
    
    return optimized_df