"""
Real-time data collection page for immediate testing
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from services.serpapi_collector import SerpAPIDataCollector, collect_real_time_data
from config.config import API_CONFIG
from data.database import DatabaseManager

def show_real_time_data_page():
    """Show real-time data collection interface"""
    
    st.title("🌐 Real-Time Product Data Collection")
    st.markdown("Collect live product data from Google Shopping using SerpAPI")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # API Status
    col1, col2 = st.columns(2)
    
    with col1:
        if API_CONFIG.get('serpapi_key'):
            st.success("✅ SerpAPI Key Configured")
        else:
            st.error("❌ SerpAPI Key Missing")
    
    with col2:
        # Show current data statistics
        stats = db_manager.get_database_stats()
        st.metric("Products in Database", stats.get('products', 0))
    
    # Quick collection section
    st.subheader("🚀 Quick Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input("Search for products:", value="iPhone 15")
        num_products = st.slider("Number of products to collect:", 1, 20, 5)
    
    with col2:
        st.write("")  # Spacing
        if st.button("🔍 Collect Products", type="primary"):
            if API_CONFIG.get('serpapi_key'):
                collect_products_manual(search_query, num_products, db_manager)
            else:
                st.error("Please configure SerpAPI key first")
    
    # Predefined categories
    st.subheader("📱 Collect by Category")
    
    categories = {
        "📱 Electronics": ["iPhone 15", "Samsung Galaxy S24", "MacBook Pro", "iPad Air"],
        "👟 Fashion": ["Nike Air Max", "Adidas Ultraboost", "Levi's jeans", "Ray-Ban sunglasses"],
        "🏠 Home": ["Dyson vacuum", "KitchenAid mixer", "Instant Pot", "Ninja blender"],
        "🎮 Gaming": ["PS5", "Xbox Series X", "Nintendo Switch", "Gaming headset"],
        "📚 Books": ["Python programming", "Machine learning", "Data science", "AI books"]
    }
    
    selected_category = st.selectbox("Choose category:", list(categories.keys()))
    
    if st.button(f"Collect {selected_category} Products"):
        if API_CONFIG.get('serpapi_key'):
            collect_category_products(categories[selected_category], db_manager)
        else:
            st.error("Please configure SerpAPI key first")
    
    # Bulk collection
    st.subheader("🔄 Bulk Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌐 Collect Popular Products", type="secondary"):
            if API_CONFIG.get('serpapi_key'):
                collect_popular_products(db_manager)
            else:
                st.error("Please configure SerpAPI key first")
    
    with col2:
        if st.button("💰 Update Existing Prices"):
            if API_CONFIG.get('serpapi_key'):
                update_existing_prices(db_manager)
            else:
                st.error("Please configure SerpAPI key first")
    
    # Recent collections
    st.subheader("📊 Recent Collections")
    show_recent_products(db_manager)

def collect_products_manual(query: str, num_products: int, db_manager: DatabaseManager):
    """Manually collect products for a specific query"""
    with st.spinner(f"Collecting {num_products} products for '{query}'..."):
        try:
            collector = SerpAPIDataCollector(API_CONFIG['serpapi_key'])
            
            # Search for products
            products = collector.search_products(query, num_products)
            
            if products:
                # Save to database
                saved_count = collector.save_products_to_database(products)
                
                st.success(f"✅ Successfully collected {len(products)} products, saved {saved_count} new ones!")
                
                # Show collected products
                if products:
                    df = pd.DataFrame(products)
                    st.dataframe(df[['name', 'price', 'brand', 'category', 'rating']])
                
            else:
                st.warning("No products found for this query. Please try a different search term.")
                
        except Exception as e:
            st.error(f"Error collecting products: {str(e)}")

def collect_category_products(product_list: list, db_manager: DatabaseManager):
    """Collect products from a predefined category"""
    with st.spinner(f"Collecting products from category..."):
        try:
            collector = SerpAPIDataCollector(API_CONFIG['serpapi_key'])
            
            all_products = []
            for product_name in product_list:
                products = collector.search_products(product_name, 3)
                all_products.extend(products)
            
            if all_products:
                saved_count = collector.save_products_to_database(all_products)
                st.success(f"✅ Collected {len(all_products)} products, saved {saved_count} new ones!")
                
                # Show summary
                df = pd.DataFrame(all_products)
                st.dataframe(df[['name', 'price', 'brand', 'category']])
            else:
                st.warning("No products found in this category.")
                
        except Exception as e:
            st.error(f"Error collecting category products: {str(e)}")

def collect_popular_products(db_manager: DatabaseManager):
    """Collect a variety of popular products"""
    with st.spinner("Collecting popular products across categories..."):
        try:
            popular_searches = [
                "iPhone 15", "Samsung Galaxy", "MacBook Pro", "iPad",
                "Nike sneakers", "Adidas shoes", "Sony headphones",
                "Dell laptop", "Canon camera", "Apple Watch"
            ]
            
            result = collect_real_time_data(API_CONFIG['serpapi_key'], popular_searches)
            
            st.success(f"✅ Bulk collection completed!")
            st.write(f"• Products collected: {result['products_collected']}")
            st.write(f"• New products saved: {result['products_saved']}")
            st.write(f"• Prices updated: {result['prices_updated']}")
            
        except Exception as e:
            st.error(f"Error in bulk collection: {str(e)}")

def update_existing_prices(db_manager: DatabaseManager):
    """Update prices for existing products"""
    with st.spinner("Updating prices for existing products..."):
        try:
            collector = SerpAPIDataCollector(API_CONFIG['serpapi_key'])
            updated_count = collector.update_existing_prices(sample_size=10)
            
            if updated_count > 0:
                st.success(f"✅ Updated prices for {updated_count} products!")
            else:
                st.info("No price updates were needed.")
                
        except Exception as e:
            st.error(f"Error updating prices: {str(e)}")

def show_recent_products(db_manager: DatabaseManager):
    """Show recently collected products"""
    try:
        products_df = db_manager.get_products(limit=10)
        
        if not products_df.empty:
            # Filter for products with real data indicators
            recent_products = products_df.head(10)
            
            st.dataframe(
                recent_products[['name', 'price', 'brand', 'category', 'rating', 'created_at']],
                use_container_width=True
            )
        else:
            st.info("No products in database yet. Start collecting some data!")
            
    except Exception as e:
        st.error(f"Error displaying recent products: {str(e)}")

if __name__ == "__main__":
    show_real_time_data_page()