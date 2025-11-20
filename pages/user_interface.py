"""
Simple user-focused interface for the e-commerce price prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from data.database import DatabaseManager
from services.simple_background_service import get_background_service
from utils.visualizations import (
    create_price_history_chart, create_price_prediction_with_history_chart,
    create_price_trends_chart
)
from utils.helpers import format_currency, format_number

def show_user_dashboard():
    """Show simplified user dashboard"""
    
    st.title("🛒 Smart Shopping Assistant")
    st.markdown("**Your AI-powered shopping companion for the best deals and price insights**")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Main user tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Product Search & Prices",
        "📈 Price Tracking", 
        "🎯 Smart Recommendations",
        "💡 Price Insights"
    ])
    
    with tab1:
        show_product_search_tab(db_manager)
    
    with tab2:
        show_price_tracking_tab(db_manager)
    
    with tab3:
        show_recommendations_tab(db_manager)
    
    with tab4:
        show_price_insights_tab(db_manager)

def show_product_search_tab(db_manager: DatabaseManager):
    """Show product search and current prices"""
    
    st.subheader("🔍 Find Products & Compare Prices")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search for products...", placeholder="e.g., iPhone, Apple, speaker, shoes")
    
    with col2:
        st.write("")  # Spacing
        search_button = st.button("🔍 Search", type="primary")
    
    # Get products from database
    try:
        products_df = db_manager.get_products()
        st.write(f"📊 **Database Status**: {len(products_df)} products available")
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return
    
    if not products_df.empty:
        categories = ['All Categories'] + sorted(products_df['category'].unique().tolist())
        
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_category = st.selectbox("Filter by Category", categories)
        with col2:
            sort_by = st.selectbox("Sort by", ["Price: Low to High", "Price: High to Low", "Rating", "Popularity"])
        
        # Start with all products
        filtered_products = products_df.copy()
        
        # Apply category filter
        if selected_category != 'All Categories':
            filtered_products = filtered_products[filtered_products['category'] == selected_category]
        
        # Apply search filter
        if search_query.strip():
            # Search in multiple fields for better results
            name_match = filtered_products['name'].str.contains(search_query, case=False, na=False)
            brand_match = filtered_products['brand'].str.contains(search_query, case=False, na=False)
            category_match = filtered_products['category'].str.contains(search_query, case=False, na=False)
            
            # Combine search criteria (OR logic)
            mask = name_match | brand_match | category_match
            filtered_products = filtered_products[mask]
        
        # Apply sorting
        if sort_by == "Price: Low to High":
            filtered_products = filtered_products.sort_values('price')
        elif sort_by == "Price: High to Low":
            filtered_products = filtered_products.sort_values('price', ascending=False)
        elif sort_by == "Rating":
            filtered_products = filtered_products.sort_values('rating', ascending=False, na_position='last')
        else:  # Popularity
            filtered_products = filtered_products.sort_values('num_reviews', ascending=False, na_position='last')
        
        # Display search results
        if not filtered_products.empty:
            st.success(f"✅ Found {len(filtered_products)} products")
            
            # Product cards
            for _, product in filtered_products.head(20).iterrows():  # Show top 20
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{product['name']}**")
                        st.write(f"*{product['category']} • {product['brand']}*")
                        if pd.notna(product['description']):
                            st.write(f"{product['description'][:100]}...")
                    
                    with col2:
                        st.metric("Price", format_currency(product['price']))
                    
                    with col3:
                        if pd.notna(product['rating']):
                            st.metric("Rating", f"{product['rating']:.1f}⭐")
                        else:
                            st.write("No rating")
                    
                    with col4:
                        if st.button(f"Track Price", key=f"track_{product['id']}"):
                            add_to_price_tracking(product['id'], product['name'])
                    
                    st.divider()
        else:
            st.info("No products found. Try a different search term or category.")
            
            # Show helpful suggestions
            with st.expander("💡 Search Suggestions"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Popular Brands Available:**")
                    popular_brands = products_df['brand'].value_counts().head(10)
                    for brand, count in popular_brands.items():
                        st.write(f"• {brand} ({count} products)")
                
                with col2:
                    st.write("**Available Categories:**")
                    categories = products_df['category'].value_counts()
                    for category, count in categories.items():
                        st.write(f"• {category} ({count} products)")
                
                st.info("💡 **Search Tips:** Try searching for brand names like 'Apple', 'iPhone', 'Dell' or categories like 'Electronics', 'Clothing'")
    else:
        st.warning("No products available. The system is initializing...")

def show_price_tracking_tab(db_manager: DatabaseManager):
    """Show price tracking for user's selected products"""
    
    st.subheader("📈 Your Price Tracking Dashboard")
    
    # Get user's tracked products (stored in session state for simplicity)
    if 'tracked_products' not in st.session_state:
        st.session_state.tracked_products = []
    
    if not st.session_state.tracked_products:
        st.info("You haven't added any products to track yet. Go to 'Product Search' to start tracking prices!")
        
        # Show some popular products to track
        st.subheader("🔥 Popular Products to Track")
        products_df = db_manager.get_products()
        
        if not products_df.empty:
            popular_products = products_df.nlargest(5, 'num_reviews')
            
            for _, product in popular_products.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{product['name']}** - {format_currency(product['price'])}")
                
                with col2:
                    st.write(f"{product['rating']:.1f}⭐")
                
                with col3:
                    if st.button("Track", key=f"popular_{product['id']}"):
                        add_to_price_tracking(product['id'], product['name'])
        return
    
    # Show tracked products
    for product_info in st.session_state.tracked_products:
        product_id, product_name = product_info['id'], product_info['name']
        
        with st.expander(f"📊 {product_name}", expanded=True):
            # Get price history
            price_history = db_manager.get_price_history(product_id)
            
            if not price_history.empty:
                # Current price info
                current_price = price_history.iloc[0]['price']
                oldest_price = price_history.iloc[-1]['price']
                price_change = current_price - oldest_price
                price_change_pct = (price_change / oldest_price) * 100 if oldest_price != 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", format_currency(current_price))
                
                with col2:
                    st.metric("Price Change", 
                             format_currency(price_change),
                             delta=f"{price_change_pct:.1f}%")
                
                with col3:
                    lowest_price = price_history['price'].min()
                    st.metric("Lowest Price", format_currency(lowest_price))
                
                with col4:
                    highest_price = price_history['price'].max()
                    st.metric("Highest Price", format_currency(highest_price))
                
                # Price history chart
                history_chart = create_price_history_chart(price_history, product_name)
                st.plotly_chart(history_chart, use_container_width=True)
                
                # Price alerts
                st.subheader("🔔 Price Alerts")
                target_price = st.number_input(
                    f"Alert me when price drops below:",
                    min_value=0.01,
                    value=float(current_price * 0.9),  # Default to 10% below current
                    step=0.01,
                    key=f"alert_{product_id}"
                )
                
                if target_price < current_price:
                    st.success(f"✅ Alert set! We'll notify you when the price drops below {format_currency(target_price)}")
                else:
                    st.info(f"Set a price below {format_currency(current_price)} to create an alert")
                
                # Remove from tracking
                if st.button(f"Stop Tracking", key=f"remove_{product_id}"):
                    remove_from_price_tracking(product_id)
            
            else:
                st.info("Collecting price data... Check back soon!")

def show_recommendations_tab(db_manager: DatabaseManager):
    """Show personalized product recommendations"""
    
    st.subheader("🎯 Personalized Recommendations")
    
    # Get user preferences
    col1, col2 = st.columns(2)
    
    with col1:
        budget_range = st.select_slider(
            "Your Budget Range",
            options=["Under $50", "$50-$100", "$100-$250", "$250-$500", "$500-$1000", "Over $1000"],
            value="$100-$250"
        )
    
    with col2:
        products_df = db_manager.get_products()
        if not products_df.empty:
            categories = products_df['category'].unique().tolist()
            preferred_categories = st.multiselect("Preferred Categories", categories, default=categories[:3])
        else:
            preferred_categories = []
    
    # Generate recommendations based on preferences
    if not products_df.empty and preferred_categories:
        # Filter by category
        recommended_products = products_df[products_df['category'].isin(preferred_categories)]
        
        # Filter by budget
        if budget_range == "Under $50":
            recommended_products = recommended_products[recommended_products['price'] < 50]
        elif budget_range == "$50-$100":
            recommended_products = recommended_products[
                (recommended_products['price'] >= 50) & (recommended_products['price'] < 100)
            ]
        elif budget_range == "$100-$250":
            recommended_products = recommended_products[
                (recommended_products['price'] >= 100) & (recommended_products['price'] < 250)
            ]
        elif budget_range == "$250-$500":
            recommended_products = recommended_products[
                (recommended_products['price'] >= 250) & (recommended_products['price'] < 500)
            ]
        elif budget_range == "$500-$1000":
            recommended_products = recommended_products[
                (recommended_products['price'] >= 500) & (recommended_products['price'] < 1000)
            ]
        else:  # Over $1000
            recommended_products = recommended_products[recommended_products['price'] >= 1000]
        
        # Sort by rating and reviews
        recommended_products = recommended_products.sort_values(['rating', 'num_reviews'], ascending=[False, False])
        
        if not recommended_products.empty:
            st.write(f"🎉 Found {len(recommended_products)} recommendations for you!")
            
            # Show top recommendations
            for _, product in recommended_products.head(10).iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{product['name']}**")
                        st.write(f"*{product['category']} • {product['brand']}*")
                        if pd.notna(product['description']):
                            st.write(f"{product['description'][:100]}...")
                    
                    with col2:
                        st.metric("Price", format_currency(product['price']))
                    
                    with col3:
                        if pd.notna(product['rating']):
                            st.metric("Rating", f"{product['rating']:.1f}⭐")
                    
                    with col4:
                        if st.button("Add to Tracking", key=f"rec_{product['id']}"):
                            add_to_price_tracking(product['id'], product['name'])
                    
                    st.divider()
        else:
            st.info("No products found in your budget range. Try adjusting your preferences.")
    
    # Show trending products
    st.subheader("🔥 Trending Products")
    
    interactions_df = db_manager.get_interactions(limit=1000)
    if not interactions_df.empty:
        # Get most interacted products
        trending_product_ids = interactions_df['product_id'].value_counts().head(5).index.tolist()
        trending_products = products_df[products_df['id'].isin(trending_product_ids)]
        
        for _, product in trending_products.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{product['name']}** - {format_currency(product['price'])}")
                st.write(f"*{product['category']}*")
            
            with col2:
                st.write(f"{product['rating']:.1f}⭐")
            
            with col3:
                if st.button("Track", key=f"trend_{product['id']}"):
                    add_to_price_tracking(product['id'], product['name'])

def show_price_insights_tab(db_manager: DatabaseManager):
    """Show price insights and market trends"""
    
    st.subheader("💡 Market Price Insights")
    
    # Market overview
    products_df = db_manager.get_products()
    
    if not products_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", format_number(len(products_df)))
        
        with col2:
            avg_price = products_df['price'].mean()
            st.metric("Average Price", format_currency(avg_price))
        
        with col3:
            categories_count = products_df['category'].nunique()
            st.metric("Categories", format_number(categories_count))
        
        with col4:
            avg_rating = products_df['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
        
        # Price trends by category
        st.subheader("📊 Price Trends by Category")
        
        price_trends = db_manager.get_price_trends(30)  # Last 30 days
        
        if not price_trends.empty:
            trends_chart = create_price_trends_chart(price_trends)
            st.plotly_chart(trends_chart, use_container_width=True)
        else:
            # Show current price distribution by category
            import plotly.express as px
            
            price_dist_chart = px.box(
                products_df,
                x='category',
                y='price',
                title="Current Price Distribution by Category"
            )
            price_dist_chart.update_xaxis(tickangle=45)
            st.plotly_chart(price_dist_chart, use_container_width=True)
        
        # Best deals section
        st.subheader("🏷️ Best Deals Right Now")
        
        # Find products with good ratings and reasonable prices
        good_deals = products_df[
            (products_df['rating'] >= 4.0) & 
            (products_df['num_reviews'] >= 10)
        ].nsmallest(5, 'price')
        
        if not good_deals.empty:
            for _, product in good_deals.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{product['name']}**")
                    st.write(f"*{product['category']} • {product['rating']:.1f}⭐ ({product['num_reviews']} reviews)*")
                
                with col2:
                    st.write(f"**{format_currency(product['price'])}**")
                
                with col3:
                    if st.button("Track Deal", key=f"deal_{product['id']}"):
                        add_to_price_tracking(product['id'], product['name'])
        
        # Category insights
        st.subheader("📈 Category Insights")
        
        category_stats = products_df.groupby('category').agg({
            'price': ['mean', 'min', 'max'],
            'rating': 'mean',
            'num_reviews': 'sum'
        }).round(2)
        
        category_stats.columns = ['Avg Price', 'Min Price', 'Max Price', 'Avg Rating', 'Total Reviews']
        category_stats['Avg Price'] = category_stats['Avg Price'].apply(lambda x: f"${x:.2f}")
        category_stats['Min Price'] = category_stats['Min Price'].apply(lambda x: f"${x:.2f}")
        category_stats['Max Price'] = category_stats['Max Price'].apply(lambda x: f"${x:.2f}")
        category_stats['Avg Rating'] = category_stats['Avg Rating'].apply(lambda x: f"{x:.1f}⭐")
        
        st.dataframe(category_stats, use_container_width=True)

def add_to_price_tracking(product_id: int, product_name: str):
    """Add a product to price tracking"""
    if 'tracked_products' not in st.session_state:
        st.session_state.tracked_products = []
    
    # Check if already tracking
    for product in st.session_state.tracked_products:
        if product['id'] == product_id:
            st.warning(f"Already tracking {product_name}!")
            return
    
    # Add to tracking
    st.session_state.tracked_products.append({
        'id': product_id,
        'name': product_name,
        'added_date': datetime.now()
    })
    
    st.success(f"✅ Added {product_name} to price tracking!")
    st.rerun()

def remove_from_price_tracking(product_id: int):
    """Remove a product from price tracking"""
    if 'tracked_products' in st.session_state:
        st.session_state.tracked_products = [
            p for p in st.session_state.tracked_products if p['id'] != product_id
        ]
        st.success("Product removed from tracking!")
        st.rerun()

def show_system_status():
    """Show system status in sidebar"""
    with st.sidebar:
        st.header("📊 System Status")
        
        service = get_background_service()
        status = service.get_status()
        
        if status['running']:
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Starting...")
        
        st.write(f"**Last Update:** {status['last_data_update']}")
        st.write(f"**Model Training:** {status['last_model_training']}")
        st.write(f"**Real Data Collection:** {status.get('last_real_data_collection', 'Never')}")
        
        st.write("**Database:**")
        st.write(f"• Products: {status['products']}")
        st.write(f"• Users: {status['users']}")
        st.write(f"• Price History: {status['price_history_points']}")

# Main interface function
def show_simple_user_interface():
    """Main function for the simple user interface"""
    
    # Start background services
    service = get_background_service()
    if not service.running:
        with st.spinner("Initializing system..."):
            service.start()
    
    # Show system status in sidebar
    show_system_status()
    
    # Show main dashboard
    show_user_dashboard()