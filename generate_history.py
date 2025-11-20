"""
Quick script to generate more price history data for better time-series training
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.database import DatabaseManager

def generate_price_history():
    """Generate price history for existing products"""
    
    db_manager = DatabaseManager()
    products_df = db_manager.get_products()
    
    print(f"Generating price history for {len(products_df)} products...")
    
    # Generate 30 days of price history for each product
    for _, product in products_df.iterrows():
        base_price = product['price']
        
        for days_ago in range(30, 0, -1):  # 30 days ago to yesterday
            # Create realistic price variations (±3% daily change)
            change_pct = random.uniform(-0.03, 0.03)
            price_variation = base_price * change_pct
            
            # Apply some seasonal trends
            if days_ago < 7:  # Recent week - slight upward trend
                price_variation += base_price * random.uniform(0, 0.01)
            
            new_price = base_price + price_variation
            
            # Keep price reasonable (within 20% of original)
            new_price = max(base_price * 0.8, min(base_price * 1.2, new_price))
            
            # Create timestamp
            timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to price history
            db_manager.add_price_history(product['id'], new_price, timestamp)
    
    # Check results
    price_history = db_manager.get_price_history()
    print(f"✅ Generated {len(price_history)} price history entries")
    
    return len(price_history)

if __name__ == "__main__":
    generate_price_history()