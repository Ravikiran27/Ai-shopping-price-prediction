"""
Simple User Interface for AI Shopping Price Prediction
Streamlit Cloud Compatible Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

def init_database():
    """Initialize SQLite database with sample data if it doesn't exist"""
    conn = sqlite3.connect('shopping_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            brand TEXT,
            price REAL,
            rating REAL,
            reviews_count INTEGER,
            availability TEXT DEFAULT 'In Stock',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            price REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')
    
    # Check if we have sample data
    cursor.execute('SELECT COUNT(*) FROM products')
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Insert sample data
        sample_products = [
            ('iPhone 15 Pro', 'Electronics', 'Apple', 999.99, 4.5, 1250, 'In Stock'),
            ('Samsung Galaxy S24', 'Electronics', 'Samsung', 899.99, 4.3, 980, 'In Stock'),
            ('MacBook Air M3', 'Electronics', 'Apple', 1299.99, 4.7, 756, 'In Stock'),
            ('Sony WH-1000XM5', 'Electronics', 'Sony', 299.99, 4.6, 2340, 'In Stock'),
            ('Nike Air Max 270', 'Fashion', 'Nike', 129.99, 4.2, 890, 'In Stock'),
            ('Adidas Ultraboost 22', 'Fashion', 'Adidas', 189.99, 4.4, 670, 'In Stock'),
            ('Canon EOS R6', 'Electronics', 'Canon', 2499.99, 4.8, 234, 'In Stock'),
            ('Dell XPS 13', 'Electronics', 'Dell', 999.99, 4.3, 445, 'In Stock'),
        ]
        
        cursor.executemany(
            'INSERT INTO products (name, category, brand, price, rating, reviews_count, availability) VALUES (?, ?, ?, ?, ?, ?, ?)',
            sample_products
        )
        
        # Add some price history
        for product_id in range(1, 9):
            base_price = cursor.execute('SELECT price FROM products WHERE id = ?', (product_id,)).fetchone()[0]
            for i in range(10):
                variation = np.random.uniform(0.9, 1.1)
                price = base_price * variation
                cursor.execute(
                    'INSERT INTO price_history (product_id, price) VALUES (?, ?)',
                    (product_id, price)
                )
    
    conn.commit()
    conn.close()

def get_products(search_query="", category="All"):
    """Get products from database"""
    conn = sqlite3.connect('shopping_data.db')
    
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND (name LIKE ? OR brand LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    if category != "All":
        query += " AND category = ?"
        params.append(category)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_price_history(product_id):
    """Get price history for a product"""
    conn = sqlite3.connect('shopping_data.db')
    query = """
        SELECT ph.price, ph.timestamp 
        FROM price_history ph
        WHERE ph.product_id = ?
        ORDER BY ph.timestamp
    """
    df = pd.read_sql_query(query, conn, params=[product_id])
    conn.close()
    return df

def predict_price(product_data):
    """Simple price prediction logic"""
    base_price = product_data['price']
    rating_factor = product_data['rating'] / 5.0
    review_factor = min(product_data['reviews_count'] / 1000, 1.0)
    
    # Simple prediction with some randomness
    prediction = base_price * (0.9 + rating_factor * 0.2 + review_factor * 0.1)
    confidence = 0.85 + rating_factor * 0.1
    
    return prediction, confidence

# Initialize database
init_database()

# Main App Layout
st.title("🛒 AI Shopping Assistant")
st.markdown("*Find the best deals with AI-powered price predictions*")

# Sidebar
with st.sidebar:
    st.header("🔍 Search & Filter")
    
    search_query = st.text_input("Search products", placeholder="e.g., iPhone, Nike shoes...")
    
    # Get categories
    conn = sqlite3.connect('shopping_data.db')
    categories = pd.read_sql_query("SELECT DISTINCT category FROM products", conn)['category'].tolist()
    conn.close()
    
    category_filter = st.selectbox("Category", ["All"] + categories)
    
    if st.button("🔍 Search", type="primary"):
        st.session_state.search_results = get_products(search_query, category_filter)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📦 Products")
    
    # Get products to display
    if st.session_state.search_results is not None:
        products_df = st.session_state.search_results
    else:
        products_df = get_products()
    
    if len(products_df) > 0:
        for idx, product in products_df.iterrows():
            with st.container():
                col_info, col_price, col_action = st.columns([3, 1, 1])
                
                with col_info:
                    st.subheader(product['name'])
                    st.write(f"**Brand:** {product['brand']}")
                    st.write(f"**Category:** {product['category']}")
                    st.write(f"⭐ {product['rating']}/5 ({product['reviews_count']} reviews)")
                
                with col_price:
                    st.metric("Current Price", f"${product['price']:.2f}")
                
                with col_action:
                    if st.button(f"📊 Analyze", key=f"analyze_{product['id']}"):
                        # Show detailed analysis
                        st.session_state.selected_product = product
                
                st.divider()
    else:
        st.info("No products found. Try different search terms.")

with col2:
    st.header("🎯 AI Insights")
    
    if 'selected_product' in st.session_state:
        product = st.session_state.selected_product
        
        st.subheader(f"Analysis: {product['name']}")
        
        # Price prediction
        predicted_price, confidence = predict_price(product)
        
        col_curr, col_pred = st.columns(2)
        with col_curr:
            st.metric("Current", f"${product['price']:.2f}")
        with col_pred:
            st.metric("Predicted", f"${predicted_price:.2f}")
        
        # Confidence
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        
        # Price history chart
        price_history = get_price_history(product['id'])
        if len(price_history) > 0:
            fig = px.line(price_history, x='timestamp', y='price', 
                         title='Price History')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if predicted_price < product['price']:
            st.success("✅ Good time to buy! Price may increase.")
        else:
            st.warning("⏳ Consider waiting. Price may decrease.")
        
        # Similar products
        similar_products = get_products(category=product['category'])
        similar_products = similar_products[similar_products['id'] != product['id']].head(3)
        
        if len(similar_products) > 0:
            st.subheader("🔄 Similar Products")
            for _, sim_product in similar_products.iterrows():
                st.write(f"• **{sim_product['name']}** - ${sim_product['price']:.2f}")
    
    else:
        st.info("👆 Select a product above to see AI analysis and predictions!")
        
        # Show some stats
        conn = sqlite3.connect('shopping_data.db')
        total_products = pd.read_sql_query("SELECT COUNT(*) as count FROM products", conn)['count'][0]
        avg_price = pd.read_sql_query("SELECT AVG(price) as avg FROM products", conn)['avg'][0]
        conn.close()
        
        st.metric("Total Products", f"{total_products:,}")
        st.metric("Average Price", f"${avg_price:.2f}")

# Footer
st.markdown("---")
st.markdown("**🤖 Powered by AI** | *Smart shopping decisions made easy*")