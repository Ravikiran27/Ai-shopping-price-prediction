"""
Test script for SerpAPI integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from services.serpapi_collector import SerpAPIDataCollector
from config.config import API_CONFIG
from data.database import DatabaseManager

def test_serpapi_collection():
    """Test SerpAPI data collection"""
    
    print("🧪 Testing SerpAPI Data Collection...")
    print(f"API Key configured: {'Yes' if API_CONFIG.get('serpapi_key') else 'No'}")
    
    if not API_CONFIG.get('serpapi_key'):
        print("❌ No API key found. Please configure SerpAPI key.")
        return
    
    try:
        # Initialize collector
        collector = SerpAPIDataCollector(API_CONFIG['serpapi_key'])
        print("✅ SerpAPI Collector initialized")
        
        # Test a simple search
        print("🔍 Searching for 'iPhone 15'...")
        products = collector.search_products("iPhone 15", 3)
        
        if products:
            print(f"✅ Found {len(products)} products:")
            for i, product in enumerate(products[:3], 1):
                print(f"  {i}. {product['name']} - ${product['price']:.2f}")
            
            # Test database saving
            db_manager = DatabaseManager()
            saved_count = collector.save_products_to_database(products)
            print(f"✅ Saved {saved_count} products to database")
            
        else:
            print("❌ No products found")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_serpapi_collection()