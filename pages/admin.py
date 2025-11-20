"""
Admin panel page for managing the e-commerce system
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.express as px

from data.database import DatabaseManager
from data.data_generator import DataGenerator, create_sample_data
from models.price_predictor import PricePredictionModel
from models.recommender import RecommendationSystem
from utils.helpers import format_currency, format_number
from utils.visualizations import create_kpi_cards

def show_admin_page():
    """Display the admin panel"""
    
    st.title("⚙️ Admin Panel")
    st.markdown("System administration and data management")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Admin tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🛍️ Product Management", 
        "👥 User Management", 
        "🤖 ML Models", 
        "💾 Data Management",
        "🔧 System Settings"
    ])
    
    with tab1:
        show_admin_dashboard(db_manager)
    
    with tab2:
        show_product_management(db_manager)
    
    with tab3:
        show_user_management(db_manager)
    
    with tab4:
        show_ml_model_management()
    
    with tab5:
        show_data_management(db_manager)
    
    with tab6:
        show_system_settings()

def show_admin_dashboard(db_manager):
    """Show admin dashboard with system overview"""
    
    st.subheader("📊 System Overview")
    
    # Get system statistics
    stats = db_manager.get_database_stats()
    
    # Calculate additional metrics
    analytics_data = db_manager.get_sales_analytics(30)  # Last 30 days
    
    # KPI Cards
    kpi_metrics = {
        "Total Products": {"value": stats.get('products', 0), "delta": "+12"},
        "Total Users": {"value": stats.get('users', 0), "delta": "+8"},
        "Total Interactions": {"value": stats.get('interactions', 0), "delta": "+156"},
        "Monthly Revenue": {"value": analytics_data['total_sales'].get('revenue', 0), "delta": "+15.2%"}
    }
    
    create_kpi_cards(kpi_metrics)
    
    st.divider()
    
    # System health indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 System Health")
        
        # Mock health indicators (in real system, these would be actual metrics)
        health_metrics = {
            "Database Connection": "✅ Healthy",
            "ML Models Status": "✅ Loaded",
            "Cache Status": "✅ Active",
            "API Response Time": "⚡ 45ms avg",
            "Memory Usage": "📊 68% (2.1GB/3GB)",
            "Disk Space": "💾 78% (156GB/200GB)"
        }
        
        for metric, status in health_metrics.items():
            st.write(f"**{metric}:** {status}")
    
    with col2:
        st.subheader("📈 Recent Activity")
        
        # Get recent interactions
        recent_query = """
            SELECT i.interaction_type, p.name as product_name, u.username, i.timestamp
            FROM interactions i
            JOIN products p ON i.product_id = p.id
            JOIN users u ON i.user_id = u.id
            ORDER BY i.timestamp DESC
            LIMIT 10
        """
        
        with db_manager.get_connection() as conn:
            recent_activity = pd.read_sql_query(recent_query, conn)
        
        if not recent_activity.empty:
            for _, activity in recent_activity.iterrows():
                timestamp = pd.to_datetime(activity['timestamp'])
                time_ago = datetime.now() - timestamp
                
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds//3600}h ago"
                else:
                    time_str = f"{time_ago.seconds//60}m ago"
                
                action_icon = {'purchase': '🛒', 'view': '👀', 'cart': '🛍️', 'rating': '⭐'}.get(activity['interaction_type'], '📝')
                
                st.write(f"{action_icon} **{activity['username']}** {activity['interaction_type']}d **{activity['product_name']}** _{time_str}_")
        else:
            st.info("No recent activity found")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Category Distribution")
        
        products_df = db_manager.get_products()
        if not products_df.empty:
            category_counts = products_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue Trend")
        
        # Mock revenue data for the last 7 days
        dates = pd.date_range(end=datetime.now(), periods=7)
        revenue_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.uniform(1000, 5000, 7).cumsum()
        })
        
        fig = px.line(revenue_data, x='Date', y='Revenue', title="Daily Revenue (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True)

def show_product_management(db_manager):
    """Show product management interface"""
    
    st.subheader("🛍️ Product Management")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Add Product"):
            st.session_state.show_add_product = True
    
    with col2:
        if st.button("📝 Edit Product"):
            st.session_state.show_edit_product = True
    
    with col3:
        if st.button("🗑️ Delete Product"):
            st.session_state.show_delete_product = True
    
    with col4:
        if st.button("📂 Import Products"):
            st.session_state.show_import_products = True
    
    # Add Product Form
    if st.session_state.get('show_add_product', False):
        with st.expander("➕ Add New Product", expanded=True):
            with st.form("add_product_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Product Name*")
                    category = st.selectbox("Category", [
                        "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
                        "Beauty", "Automotive", "Toys", "Health", "Food"
                    ])
                    brand = st.text_input("Brand")
                    price = st.number_input("Price ($)*", min_value=0.01, value=10.0)
                
                with col2:
                    description = st.text_area("Description")
                    rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
                    num_reviews = st.number_input("Number of Reviews", min_value=0, value=0)
                
                # Features
                st.subheader("Product Features")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color = st.text_input("Color", value="Black")
                    size = st.text_input("Size", value="Medium")
                
                with col2:
                    weight = st.text_input("Weight", value="1.0 kg")
                    material = st.text_input("Material", value="Plastic")
                
                with col3:
                    warranty = st.text_input("Warranty", value="1 year")
                    in_stock = st.checkbox("In Stock", value=True)
                
                if st.form_submit_button("Add Product"):
                    if name and price:
                        try:
                            product_data = {
                                'name': name,
                                'category': category,
                                'brand': brand,
                                'price': price,
                                'description': description,
                                'rating': rating,
                                'num_reviews': num_reviews,
                                'features': {
                                    'color': color,
                                    'size': size,
                                    'weight': weight,
                                    'material': material,
                                    'warranty': warranty,
                                    'in_stock': in_stock
                                }
                            }
                            
                            product_id = db_manager.add_product(product_data)
                            st.success(f"Product added successfully! ID: {product_id}")
                            st.session_state.show_add_product = False
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error adding product: {str(e)}")
                    else:
                        st.error("Please fill in all required fields (*)")
    
    # Import Products
    if st.session_state.get('show_import_products', False):
        with st.expander("📂 Import Products from CSV", expanded=True):
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview:")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("Import Products"):
                        success_count = 0
                        error_count = 0
                        
                        for _, row in df.iterrows():
                            try:
                                product_data = {
                                    'name': row.get('name', 'Unknown Product'),
                                    'category': row.get('category', 'Other'),
                                    'brand': row.get('brand', 'Unknown'),
                                    'price': float(row.get('price', 0)),
                                    'description': row.get('description', ''),
                                    'rating': float(row.get('rating', 4.0)),
                                    'num_reviews': int(row.get('num_reviews', 0)),
                                    'features': {}
                                }
                                
                                db_manager.add_product(product_data)
                                success_count += 1
                            
                            except Exception as e:
                                error_count += 1
                                st.warning(f"Error importing row: {str(e)}")
                        
                        st.success(f"Import completed! {success_count} products added, {error_count} errors")
                        st.session_state.show_import_products = False
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    # Product List
    st.subheader("📋 Product List")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search products")
    
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + [
            "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
            "Beauty", "Automotive", "Toys", "Health", "Food"
        ])
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Name", "Price", "Rating", "Date Added"])
    
    # Get and display products
    products_df = db_manager.get_products()
    
    if not products_df.empty:
        # Apply filters
        if search_term:
            products_df = products_df[products_df['name'].str.contains(search_term, case=False, na=False)]
        
        if category_filter and category_filter != "All":
            products_df = products_df[products_df['category'] == category_filter]
        
        # Sort
        if sort_by == "Price":
            products_df = products_df.sort_values('price', ascending=False)
        elif sort_by == "Rating":
            products_df = products_df.sort_values('rating', ascending=False)
        elif sort_by == "Date Added":
            products_df = products_df.sort_values('created_at', ascending=False)
        else:
            products_df = products_df.sort_values('name')
        
        # Display products
        st.write(f"Found {len(products_df)} products")
        
        # Pagination
        page_size = 20
        total_pages = (len(products_df) - 1) // page_size + 1
        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_products = products_df.iloc[start_idx:end_idx]
        
        # Display as cards
        for i in range(0, len(page_products), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(page_products):
                    product = page_products.iloc[i + j]
                    
                    with col:
                        with st.container():
                            st.write(f"**{product['name']}**")
                            st.write(f"Category: {product['category']}")
                            st.write(f"Brand: {product['brand']}")
                            st.write(f"Price: {format_currency(product['price'])}")
                            st.write(f"Rating: {product.get('rating', 0):.1f} ⭐")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Edit", key=f"edit_{product['id']}"):
                                    st.session_state.edit_product_id = product['id']
                            with col2:
                                if st.button("Delete", key=f"delete_{product['id']}"):
                                    try:
                                        db_manager.delete_product(product['id'])
                                        st.success("Product deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
    
    else:
        st.info("No products found. Add some products to get started!")

def show_user_management(db_manager):
    """Show user management interface"""
    
    st.subheader("👥 User Management")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add User"):
            st.session_state.show_add_user = True
    
    with col2:
        if st.button("📊 User Analytics"):
            st.session_state.show_user_analytics = True
    
    with col3:
        if st.button("📂 Export Users"):
            users_df = db_manager.get_users()
            if not users_df.empty:
                csv = users_df.to_csv(index=False)
                st.download_button(
                    label="Download Users CSV",
                    data=csv,
                    file_name=f"users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Add User Form
    if st.session_state.get('show_add_user', False):
        with st.expander("➕ Add New User", expanded=True):
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username*")
                    email = st.text_input("Email*")
                    age = st.number_input("Age", min_value=13, max_value=120, value=25)
                
                with col2:
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    preferred_categories = st.multiselect("Preferred Categories", [
                        "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
                        "Beauty", "Automotive", "Toys", "Health", "Food"
                    ])
                
                if st.form_submit_button("Add User"):
                    if username and email:
                        try:
                            user_data = {
                                'username': username,
                                'email': email,
                                'age': age,
                                'gender': gender,
                                'preferences': {
                                    'preferred_categories': preferred_categories,
                                    'price_sensitivity': 'Medium',
                                    'brand_loyalty': 'Medium'
                                }
                            }
                            
                            user_id = db_manager.add_user(user_data)
                            st.success(f"User added successfully! ID: {user_id}")
                            st.session_state.show_add_user = False
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error adding user: {str(e)}")
                    else:
                        st.error("Please fill in all required fields (*)")
    
    # User List
    st.subheader("📋 User List")
    
    users_df = db_manager.get_users()
    
    if not users_df.empty:
        # Search and filters
        col1, col2 = st.columns(2)
        
        with col1:
            search_user = st.text_input("🔍 Search users")
        
        with col2:
            sort_by_user = st.selectbox("Sort by", ["Username", "Email", "Age", "Join Date"])
        
        # Apply filters
        if search_user:
            users_df = users_df[
                users_df['username'].str.contains(search_user, case=False, na=False) |
                users_df['email'].str.contains(search_user, case=False, na=False)
            ]
        
        # Display users table
        display_columns = ['id', 'username', 'email', 'age', 'gender', 'created_at']
        st.dataframe(users_df[display_columns], use_container_width=True)
        
        # User Analytics
        if st.session_state.get('show_user_analytics', False):
            with st.expander("📊 User Analytics", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution
                    if 'age' in users_df.columns:
                        age_counts = users_df['age'].value_counts().sort_index()
                        fig = px.histogram(users_df, x='age', title="Age Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Gender distribution
                    if 'gender' in users_df.columns:
                        gender_counts = users_df['gender'].value_counts()
                        fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No users found. Add some users to get started!")

def show_ml_model_management():
    """Show ML model management interface"""
    
    st.subheader("🤖 Machine Learning Models")
    
    # Model status
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔮 Price Prediction Models")
        
        # Initialize price model if not exists
        if 'price_model' not in st.session_state:
            st.session_state.price_model = PricePredictionModel()
        
        price_model = st.session_state.price_model
        
        # Model status
        if price_model.model_performance:
            st.success("✅ Models are trained and ready")
            
            # Show performance
            comparison_df = price_model.get_model_comparison()
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
                
                best_model = comparison_df.loc[comparison_df['Test RMSE'].idxmin(), 'Model']
                st.info(f"🏆 Best model: {best_model}")
        else:
            st.warning("⚠️ Models not trained yet")
        
        # Model actions
        if st.button("🏋️ Train Price Models"):
            with st.spinner("Training models..."):
                try:
                    db_manager = st.session_state.db_manager
                    products_df = db_manager.get_products()
                    
                    if len(products_df) >= 50:
                        X, y = price_model.prepare_data(products_df)
                        results = price_model.train_models(X, y)
                        price_model.save_models()
                        st.success("Price prediction models trained successfully!")
                    else:
                        st.error("Need at least 50 products to train models")
                
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
        
        if st.button("💾 Save Price Models"):
            try:
                price_model.save_models()
                st.success("Models saved successfully!")
            except Exception as e:
                st.error(f"Error saving models: {str(e)}")
        
        if st.button("📂 Load Price Models"):
            try:
                price_model.load_models()
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
    
    with col2:
        st.write("### 🎯 Recommendation Models")
        
        # Initialize recommendation system if not exists
        if 'recommendation_system' not in st.session_state:
            st.session_state.recommendation_system = RecommendationSystem()
        
        rec_system = st.session_state.recommendation_system
        
        # Model status
        if hasattr(rec_system, 'collaborative_model') and rec_system.collaborative_model is not None:
            st.success("✅ Recommendation models are ready")
        else:
            st.warning("⚠️ Recommendation models not trained yet")
        
        # Model actions
        if st.button("🏋️ Train Recommendation Models"):
            with st.spinner("Training recommendation models..."):
                try:
                    db_manager = st.session_state.db_manager
                    
                    # Get data
                    interactions_query = """
                        SELECT user_id, product_id, interaction_type, rating, timestamp
                        FROM interactions
                    """
                    
                    with db_manager.get_connection() as conn:
                        interactions_df = pd.read_sql_query(interactions_query, conn)
                    
                    if len(interactions_df) >= 100:
                        products_df = db_manager.get_products()
                        users_df = db_manager.get_users()
                        
                        rec_system.prepare_data(interactions_df, products_df, users_df)
                        rec_system.train_collaborative_filtering()
                        rec_system.train_content_based(products_df)
                        rec_system.save_models()
                        
                        st.success("Recommendation models trained successfully!")
                    else:
                        st.error("Need at least 100 interactions to train models")
                
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
        
        if st.button("💾 Save Recommendation Models"):
            try:
                rec_system.save_models()
                st.success("Recommendation models saved successfully!")
            except Exception as e:
                st.error(f"Error saving models: {str(e)}")
        
        if st.button("📂 Load Recommendation Models"):
            try:
                rec_system.load_models()
                st.success("Recommendation models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
    
    # Model performance monitoring
    st.divider()
    st.subheader("📊 Model Performance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Prediction Performance**")
        
        # Mock performance metrics over time
        dates = pd.date_range(end=datetime.now(), periods=30)
        performance_data = pd.DataFrame({
            'Date': dates,
            'RMSE': np.random.uniform(10, 50, 30),
            'R2_Score': np.random.uniform(0.7, 0.95, 30)
        })
        
        fig = px.line(performance_data, x='Date', y='RMSE', title='RMSE Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Recommendation Performance**")
        
        # Mock recommendation metrics
        rec_performance = pd.DataFrame({
            'Date': dates,
            'Precision': np.random.uniform(0.1, 0.3, 30),
            'Recall': np.random.uniform(0.15, 0.35, 30)
        })
        
        fig = px.line(rec_performance, x='Date', y=['Precision', 'Recall'], title='Recommendation Metrics Over Time')
        st.plotly_chart(fig, use_container_width=True)

def show_data_management(db_manager):
    """Show data management interface"""
    
    st.subheader("💾 Data Management")
    
    # Data statistics
    stats = db_manager.get_database_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products", format_number(stats.get('products', 0)))
    
    with col2:
        st.metric("Users", format_number(stats.get('users', 0)))
    
    with col3:
        st.metric("Interactions", format_number(stats.get('interactions', 0)))
    
    with col4:
        st.metric("Categories", format_number(stats.get('categories', 0)))
    
    st.divider()
    
    # Data operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Generate Sample Data")
        
        sample_products = st.number_input("Number of Products", min_value=10, max_value=10000, value=1000)
        sample_users = st.number_input("Number of Users", min_value=10, max_value=5000, value=500)
        sample_interactions = st.number_input("Number of Interactions", min_value=100, max_value=50000, value=5000)
        
        if st.button("🎲 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                try:
                    # Clear existing data first
                    if st.checkbox("Clear existing data first"):
                        db_manager.clear_all_data()
                    
                    # Generate new data
                    stats = create_sample_data()
                    st.success(f"Sample data generated successfully! Created {stats['products']} products, {stats['users']} users, and {stats['interactions']} interactions.")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error generating sample data: {str(e)}")
        
        st.write("### 🗑️ Data Cleanup")
        
        if st.button("🧹 Clean Old Data"):
            with st.spinner("Cleaning old data..."):
                # Mock cleanup operation
                st.success("Old data cleaned successfully!")
        
        if st.button("⚠️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                try:
                    db_manager.clear_all_data()
                    st.success("All data cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
    
    with col2:
        st.write("### 💾 Backup & Restore")
        
        if st.button("📦 Create Backup"):
            try:
                backup_path = db_manager.backup_database()
                st.success(f"Backup created successfully: {backup_path}")
            except Exception as e:
                st.error(f"Error creating backup: {str(e)}")
        
        st.write("### 📁 Data Export")
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Products"):
                products_df = db_manager.get_products()
                if not products_df.empty:
                    if export_format == "CSV":
                        csv = products_df.to_csv(index=False)
                        st.download_button(
                            label="Download Products CSV",
                            data=csv,
                            file_name=f"products_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:  # JSON
                        json_data = products_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="Download Products JSON",
                            data=json_data,
                            file_name=f"products_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
        with col2:
            if st.button("👥 Export Users"):
                users_df = db_manager.get_users()
                if not users_df.empty:
                    if export_format == "CSV":
                        csv = users_df.to_csv(index=False)
                        st.download_button(
                            label="Download Users CSV",
                            data=csv,
                            file_name=f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

def show_system_settings():
    """Show system settings interface"""
    
    st.subheader("🔧 System Settings")
    
    # Application settings
    with st.expander("⚙️ Application Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**General Settings**")
            app_name = st.text_input("Application Name", value="E-Commerce AI System")
            max_recommendations = st.number_input("Max Recommendations", min_value=5, max_value=50, value=20)
            cache_timeout = st.number_input("Cache Timeout (seconds)", min_value=60, max_value=3600, value=300)
        
        with col2:
            st.write("**ML Model Settings**")
            model_retrain_interval = st.number_input("Model Retrain Interval (days)", min_value=1, max_value=30, value=7)
            prediction_confidence_threshold = st.slider("Prediction Confidence Threshold", 0.0, 1.0, 0.8, 0.05)
            enable_auto_training = st.checkbox("Enable Auto Training", value=True)
    
    # Database settings
    with st.expander("🗄️ Database Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Connection Settings**")
            db_backup_interval = st.number_input("Backup Interval (hours)", min_value=1, max_value=168, value=24)
            max_connections = st.number_input("Max Connections", min_value=1, max_value=100, value=20)
        
        with col2:
            st.write("**Performance Settings**")
            query_timeout = st.number_input("Query Timeout (seconds)", min_value=5, max_value=300, value=30)
            enable_query_logging = st.checkbox("Enable Query Logging", value=False)
    
    # Security settings
    with st.expander("🔒 Security Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Authentication**")
            enable_authentication = st.checkbox("Enable Authentication", value=False)
            session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=480, value=60)
        
        with col2:
            st.write("**API Settings**")
            api_rate_limit = st.number_input("API Rate Limit (requests/minute)", min_value=10, max_value=1000, value=100)
            enable_api_logging = st.checkbox("Enable API Logging", value=True)
    
    # Monitoring settings
    with st.expander("📊 Monitoring Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Monitoring**")
            enable_performance_monitoring = st.checkbox("Enable Performance Monitoring", value=True)
            monitoring_interval = st.number_input("Monitoring Interval (seconds)", min_value=10, max_value=300, value=60)
        
        with col2:
            st.write("**Alerting**")
            enable_alerts = st.checkbox("Enable Alerts", value=True)
            alert_email = st.text_input("Alert Email", value="admin@example.com")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        settings = {
            'app_name': app_name,
            'max_recommendations': max_recommendations,
            'cache_timeout': cache_timeout,
            'model_retrain_interval': model_retrain_interval,
            'prediction_confidence_threshold': prediction_confidence_threshold,
            'enable_auto_training': enable_auto_training,
            'db_backup_interval': db_backup_interval,
            'max_connections': max_connections,
            'query_timeout': query_timeout,
            'enable_query_logging': enable_query_logging,
            'enable_authentication': enable_authentication,
            'session_timeout': session_timeout,
            'api_rate_limit': api_rate_limit,
            'enable_api_logging': enable_api_logging,
            'enable_performance_monitoring': enable_performance_monitoring,
            'monitoring_interval': monitoring_interval,
            'enable_alerts': enable_alerts,
            'alert_email': alert_email
        }
        
        # In a real application, these settings would be saved to a configuration file
        st.success("Settings saved successfully!")
        st.json(settings)
    
    # System information
    st.divider()
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Application Info**")
        st.write(f"Version: 1.0.0")
        st.write(f"Python: 3.9+")
        st.write(f"Streamlit: 1.28+")
    
    with col2:
        st.write("**Database Info**")
        st.write(f"Type: SQLite")
        st.write(f"Size: ~15MB")  # Mock data
        st.write(f"Tables: 6")
    
    with col3:
        st.write("**System Resources**")
        st.write(f"Memory Usage: 450MB")  # Mock data
        st.write(f"CPU Usage: 12%")  # Mock data
        st.write(f"Disk Space: 2.1GB")  # Mock data

if __name__ == "__main__":
    show_admin_page()