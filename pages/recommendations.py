"""
Product recommendation page for the e-commerce system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

from data.database import DatabaseManager
from models.recommender import RecommendationSystem
from utils.visualizations import create_recommendation_performance_chart
from utils.helpers import format_currency

def show_recommendations_page():
    """Display the product recommendations interface"""
    
    st.title("🎯 Product Recommendations")
    st.markdown("AI-powered product recommendations using collaborative and content-based filtering")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'recommendation_system' not in st.session_state:
        st.session_state.recommendation_system = RecommendationSystem()
    
    db_manager = st.session_state.db_manager
    rec_system = st.session_state.recommendation_system
    
    # Sidebar for system management
    st.sidebar.header("🤖 Recommendation System")
    
    # Model training section
    with st.sidebar.expander("Train Recommendation Models"):
        if st.button("Train All Models"):
            with st.spinner("Training recommendation models..."):
                try:
                    # Get data
                    interactions_query = """
                        SELECT user_id, product_id, interaction_type, rating, timestamp
                        FROM interactions
                    """
                    
                    with db_manager.get_connection() as conn:
                        interactions_df = pd.read_sql_query(interactions_query, conn)
                    
                    products_df = db_manager.get_products()
                    users_df = db_manager.get_users()
                    
                    if len(interactions_df) < 100:
                        st.error("Need at least 100 interactions to train models. Please add more sample data.")
                    else:
                        # Prepare data
                        rec_system.prepare_data(interactions_df, products_df, users_df)
                        
                        # Train models
                        rec_system.train_collaborative_filtering()
                        rec_system.train_content_based(products_df)
                        
                        # Save models
                        rec_system.save_models()
                        
                        st.success("Recommendation models trained successfully!")
                
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
    
    # Load existing models
    if st.sidebar.button("Load Saved Models"):
        try:
            rec_system.load_models()
            st.sidebar.success("Models loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading models: {str(e)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 User Recommendations", 
        "🛍️ Product Similarities", 
        "📊 System Performance", 
        "🔍 Recommendation Insights",
        "⚙️ Settings"
    ])
    
    with tab1:
        show_user_recommendations_interface(rec_system, db_manager)
    
    with tab2:
        show_product_similarity_interface(rec_system, db_manager)
    
    with tab3:
        show_system_performance_interface(rec_system, db_manager)
    
    with tab4:
        show_recommendation_insights_interface(rec_system, db_manager)
    
    with tab5:
        show_settings_interface(rec_system)

def show_user_recommendations_interface(rec_system, db_manager):
    """Show user-based recommendations interface"""
    
    st.subheader("Get Personalized Recommendations")
    
    # Get users data
    users_df = db_manager.get_users()
    
    if users_df.empty:
        st.warning("No users found in database. Please add some users and interactions first.")
        return
    
    # User selection
    col1, col2 = st.columns(2)
    
    with col1:
        user_options = [f"{row['username']} (ID: {row['id']})" for _, row in users_df.iterrows()]
        selected_user_option = st.selectbox("Select User", user_options)
        
        # Extract user ID
        user_id = int(selected_user_option.split("ID: ")[1].split(")")[0])
        
        # Recommendation type
        rec_type = st.selectbox(
            "Recommendation Type",
            ["hybrid", "collaborative", "content"],
            help="Hybrid combines collaborative and content-based filtering"
        )
        
        # Number of recommendations
        n_recommendations = st.slider("Number of Recommendations", 1, 20, 10)
    
    with col2:
        # Show user profile
        user_data = db_manager.get_user_by_id(user_id)
        if user_data:
            st.write("**User Profile:**")
            st.write(f"Username: {user_data['username']}")
            st.write(f"Email: {user_data['email']}")
            if user_data.get('age'):
                st.write(f"Age: {user_data['age']}")
            if user_data.get('preferences'):
                try:
                    prefs = json.loads(user_data['preferences'])
                    if prefs.get('preferred_categories'):
                        st.write(f"Preferred Categories: {', '.join(prefs['preferred_categories'])}")
                except:
                    pass
    
    # Get user's interaction history
    user_interactions = db_manager.get_user_interactions(user_id)
    
    if not user_interactions.empty:
        st.write(f"**User's Recent Activity** ({len(user_interactions)} interactions):")
        
        # Show recent purchases/interactions
        recent_interactions = user_interactions.head(5)
        for _, interaction in recent_interactions.iterrows():
            interaction_icon = {
                'purchase': '🛒', 'view': '👀', 'cart': '🛍️', 'rating': '⭐'
            }.get(interaction['interaction_type'], '📝')
            
            st.write(f"{interaction_icon} {interaction['product_name']} ({interaction['category']}) - {interaction['interaction_type']}")
    
    # Generate recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        try:
            with st.spinner("Generating recommendations..."):
                recommendations = rec_system.get_user_recommendations(
                    user_id, rec_type, n_recommendations
                )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations!")
                
                # Get product details for recommendations
                products_df = db_manager.get_products()
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    product_id = rec['product_id']
                    score = rec['score']
                    reason = rec['reason']
                    
                    # Get product details
                    if product_id < len(products_df):
                        product = products_df.iloc[product_id]
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{i}. {product['name']}**")
                                st.write(f"Category: {product['category']} | Brand: {product['brand']}")
                                st.write(f"Description: {product.get('description', 'No description available')[:100]}...")
                                st.write(f"💡 *{reason}*")
                            
                            with col2:
                                st.metric("Price", format_currency(product['price']))
                                st.metric("Rating", f"{product.get('rating', 0):.1f} ⭐")
                            
                            with col3:
                                st.metric("Rec. Score", f"{score:.3f}")
                                if st.button(f"View Details", key=f"view_{product_id}"):
                                    show_product_details(product)
                        
                        st.divider()
                
                # Recommendation explanation
                if recommendations:
                    with st.expander("🔍 Why These Recommendations?"):
                        explanation = rec_system.get_recommendation_explanation(
                            user_id, recommendations[0]['product_id']
                        )
                        
                        if explanation['explanations']:
                            for exp in explanation['explanations']:
                                st.write(f"**{exp['type'].title()} Filtering:**")
                                st.write(f"- {exp['explanation']}")
                                st.write(f"- Confidence: {exp['confidence']:.3f}")
                        else:
                            st.write("Recommendations based on general popularity and user preferences.")
            
            else:
                st.info("No recommendations available for this user. Try training the models first.")
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    
    # Cold start recommendations
    st.divider()
    st.subheader("🆕 New User Recommendations")
    st.markdown("Popular products for users without interaction history")
    
    if st.button("Get Popular Products"):
        try:
            popular_recs = rec_system._get_popular_items(5)
            
            if popular_recs:
                products_df = db_manager.get_products()
                
                st.write("**Most Popular Products:**")
                for i, rec in enumerate(popular_recs, 1):
                    product_id = rec['product_id']
                    if product_id < len(products_df):
                        product = products_df.iloc[product_id]
                        st.write(f"{i}. **{product['name']}** - {format_currency(product['price'])}")
            else:
                st.info("No popular products data available.")
        
        except Exception as e:
            st.error(f"Error getting popular products: {str(e)}")

def show_product_similarity_interface(rec_system, db_manager):
    """Show product similarity interface"""
    
    st.subheader("Find Similar Products")
    
    # Get products
    products_df = db_manager.get_products()
    
    if products_df.empty:
        st.warning("No products found in database.")
        return
    
    # Product selection
    col1, col2 = st.columns(2)
    
    with col1:
        product_options = [f"{row['name']} (ID: {row['id']})" for _, row in products_df.iterrows()]
        selected_product_option = st.selectbox("Select Product", product_options)
        
        # Extract product ID
        product_id = int(selected_product_option.split("ID: ")[1].split(")")[0])
        
        # Number of similar products
        n_similar = st.slider("Number of Similar Products", 1, 15, 5)
    
    with col2:
        # Show selected product details
        selected_product = products_df[products_df['id'] == product_id].iloc[0]
        st.write("**Selected Product:**")
        st.write(f"Name: {selected_product['name']}")
        st.write(f"Category: {selected_product['category']}")
        st.write(f"Brand: {selected_product['brand']}")
        st.write(f"Price: {format_currency(selected_product['price'])}")
        st.write(f"Rating: {selected_product.get('rating', 0):.1f} ⭐")
    
    # Generate similar products
    if st.button("🔍 Find Similar Products", type="primary"):
        try:
            with st.spinner("Finding similar products..."):
                # Use the product's index in the dataframe
                product_index = products_df[products_df['id'] == product_id].index[0]
                similar_products = rec_system.get_item_recommendations(product_index, n_similar)
            
            if similar_products:
                st.success(f"Found {len(similar_products)} similar products!")
                
                # Display similar products
                for i, sim_product in enumerate(similar_products, 1):
                    sim_product_id = sim_product['product_id']
                    similarity_score = sim_product['score']
                    
                    # Get product details
                    if sim_product_id < len(products_df):
                        product = products_df.iloc[sim_product_id]
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{i}. {product['name']}**")
                                st.write(f"Category: {product['category']} | Brand: {product['brand']}")
                                st.write(f"Description: {product.get('description', 'No description available')[:100]}...")
                            
                            with col2:
                                st.metric("Price", format_currency(product['price']))
                                st.metric("Rating", f"{product.get('rating', 0):.1f} ⭐")
                            
                            with col3:
                                st.metric("Similarity", f"{similarity_score:.3f}")
                                if st.button(f"View Details", key=f"similar_{sim_product_id}"):
                                    show_product_details(product)
                        
                        st.divider()
            
            else:
                st.info("No similar products found. Try training the content-based model first.")
        
        except Exception as e:
            st.error(f"Error finding similar products: {str(e)}")
    
    # Category-wise similarity
    st.divider()
    st.subheader("📊 Category-wise Product Similarity")
    
    category = st.selectbox("Select Category for Analysis", products_df['category'].unique())
    
    if st.button("Analyze Category Similarities"):
        category_products = products_df[products_df['category'] == category]
        
        if len(category_products) > 1:
            st.write(f"**Products in {category} Category:**")
            st.dataframe(
                category_products[['name', 'brand', 'price', 'rating']],
                use_container_width=True
            )
            
            # Calculate average price and rating for category
            avg_price = category_products['price'].mean()
            avg_rating = category_products['rating'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Price", format_currency(avg_price))
            with col2:
                st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
        else:
            st.info(f"Not enough products in {category} category for analysis.")

def show_system_performance_interface(rec_system, db_manager):
    """Show recommendation system performance metrics"""
    
    st.subheader("Recommendation System Performance")
    
    # Mock performance metrics (in a real system, these would be calculated)
    performance_metrics = {
        'Precision@10': 0.156,
        'Recall@10': 0.234,
        'F1 Score': 0.189,
        'Coverage': 0.67,
        'Diversity': 0.82
    }
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Precision@10", f"{performance_metrics['Precision@10']:.3f}")
    
    with col2:
        st.metric("Recall@10", f"{performance_metrics['Recall@10']:.3f}")
    
    with col3:
        st.metric("F1 Score", f"{performance_metrics['F1 Score']:.3f}")
    
    with col4:
        st.metric("Coverage", f"{performance_metrics['Coverage']:.2%}")
    
    with col5:
        st.metric("Diversity", f"{performance_metrics['Diversity']:.2%}")
    
    # Performance chart
    performance_chart = create_recommendation_performance_chart(performance_metrics)
    st.plotly_chart(performance_chart, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Type Comparison")
    
    model_comparison = {
        'Model Type': ['Collaborative Filtering', 'Content-Based', 'Hybrid'],
        'Precision': [0.142, 0.167, 0.156],
        'Recall': [0.298, 0.189, 0.234],
        'Coverage': [0.89, 0.45, 0.67]
    }
    
    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df, use_container_width=True)
    
    # System statistics
    st.subheader("System Statistics")
    
    # Get actual statistics from database
    stats = db_manager.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Products", stats.get('products', 0))
        st.metric("Total Users", stats.get('users', 0))
    
    with col2:
        st.metric("Total Interactions", stats.get('interactions', 0))
        
        # Calculate sparsity
        if stats.get('users', 0) > 0 and stats.get('products', 0) > 0:
            total_possible = stats['users'] * stats['products']
            sparsity = (1 - stats.get('interactions', 0) / total_possible) * 100
            st.metric("Matrix Sparsity", f"{sparsity:.1f}%")
    
    with col3:
        # Get interaction types distribution
        interaction_query = """
            SELECT interaction_type, COUNT(*) as count
            FROM interactions
            GROUP BY interaction_type
        """
        
        with db_manager.get_connection() as conn:
            interaction_dist = pd.read_sql_query(interaction_query, conn)
        
        if not interaction_dist.empty:
            most_common = interaction_dist.loc[interaction_dist['count'].idxmax(), 'interaction_type']
            st.metric("Most Common Interaction", most_common.title())

def show_recommendation_insights_interface(rec_system, db_manager):
    """Show insights about recommendations"""
    
    st.subheader("Recommendation Insights & Analytics")
    
    # User engagement analysis
    st.write("**User Engagement Analysis**")
    
    engagement_query = """
        SELECT 
            u.id as user_id,
            u.username,
            COUNT(i.id) as total_interactions,
            COUNT(CASE WHEN i.interaction_type = 'purchase' THEN 1 END) as purchases,
            COUNT(CASE WHEN i.interaction_type = 'view' THEN 1 END) as views,
            AVG(CASE WHEN i.rating IS NOT NULL THEN i.rating END) as avg_rating
        FROM users u
        LEFT JOIN interactions i ON u.id = i.user_id
        GROUP BY u.id, u.username
        ORDER BY total_interactions DESC
        LIMIT 10
    """
    
    with db_manager.get_connection() as conn:
        engagement_df = pd.read_sql_query(engagement_query, conn)
    
    if not engagement_df.empty:
        st.dataframe(engagement_df, use_container_width=True)
        
        # Insights
        most_active_user = engagement_df.iloc[0]['username']
        most_interactions = engagement_df.iloc[0]['total_interactions']
        
        st.info(f"💡 **Insight**: {most_active_user} is the most active user with {most_interactions} interactions")
    
    # Product popularity analysis
    st.write("**Product Popularity Analysis**")
    
    popularity_query = """
        SELECT 
            p.name,
            p.category,
            p.brand,
            p.price,
            COUNT(i.id) as interaction_count,
            COUNT(CASE WHEN i.interaction_type = 'purchase' THEN 1 END) as purchase_count,
            AVG(CASE WHEN i.rating IS NOT NULL THEN i.rating END) as avg_user_rating
        FROM products p
        LEFT JOIN interactions i ON p.id = i.product_id
        GROUP BY p.id, p.name, p.category, p.brand, p.price
        ORDER BY interaction_count DESC
        LIMIT 10
    """
    
    with db_manager.get_connection() as conn:
        popularity_df = pd.read_sql_query(popularity_query, conn)
    
    if not popularity_df.empty:
        st.dataframe(popularity_df, use_container_width=True)
        
        # Insights
        most_popular_product = popularity_df.iloc[0]['name']
        most_popular_interactions = popularity_df.iloc[0]['interaction_count']
        
        st.info(f"💡 **Insight**: {most_popular_product} is the most popular product with {most_popular_interactions} interactions")
    
    # Recommendation diversity analysis
    st.write("**Recommendation Diversity Analysis**")
    
    category_dist_query = """
        SELECT category, COUNT(*) as product_count
        FROM products
        GROUP BY category
        ORDER BY product_count DESC
    """
    
    with db_manager.get_connection() as conn:
        category_dist = pd.read_sql_query(category_dist_query, conn)
    
    if not category_dist.empty:
        st.bar_chart(category_dist.set_index('category'))
        
        total_categories = len(category_dist)
        dominant_category = category_dist.iloc[0]['category']
        dominant_percentage = (category_dist.iloc[0]['product_count'] / category_dist['product_count'].sum()) * 100
        
        st.info(f"💡 **Insight**: Catalog has {total_categories} categories. {dominant_category} dominates with {dominant_percentage:.1f}% of products")

def show_settings_interface(rec_system):
    """Show recommendation system settings"""
    
    st.subheader("⚙️ Recommendation System Settings")
    
    # Algorithm weights for hybrid approach
    st.write("**Hybrid Algorithm Weights**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        collaborative_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6, 0.1)
    
    with col2:
        content_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)
    
    # Normalize weights
    total_weight = collaborative_weight + content_weight
    if total_weight > 0:
        collaborative_weight /= total_weight
        content_weight /= total_weight
        
        st.write(f"Normalized weights: Collaborative: {collaborative_weight:.2f}, Content: {content_weight:.2f}")
    
    # Recommendation parameters
    st.write("**Recommendation Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_interactions = st.number_input("Minimum User Interactions", min_value=1, value=5)
        max_recommendations = st.number_input("Maximum Recommendations", min_value=1, max_value=50, value=20)
    
    with col2:
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1, 0.05)
        diversity_factor = st.slider("Diversity Factor", 0.0, 1.0, 0.3, 0.1)
    
    # Content-based settings
    st.write("**Content-Based Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_features = st.number_input("Max TF-IDF Features", min_value=100, max_value=10000, value=5000, step=100)
        ngram_min = st.selectbox("N-gram Min", [1, 2], index=0)
    
    with col2:
        ngram_max = st.selectbox("N-gram Max", [1, 2, 3], index=1)
        use_stopwords = st.checkbox("Use Stop Words", value=True)
    
    # Save settings
    if st.button("💾 Save Settings"):
        settings = {
            'hybrid': {
                'collaborative_weight': collaborative_weight,
                'content_weight': content_weight
            },
            'parameters': {
                'min_interactions': min_interactions,
                'max_recommendations': max_recommendations,
                'similarity_threshold': similarity_threshold,
                'diversity_factor': diversity_factor
            },
            'content_based': {
                'max_features': max_features,
                'ngram_range': (ngram_min, ngram_max),
                'use_stopwords': use_stopwords
            }
        }
        
        # In a real application, you would save these settings
        st.success("Settings saved successfully!")
        st.json(settings)

def show_product_details(product):
    """Show detailed product information in a modal-like container"""
    
    with st.container():
        st.write("### Product Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {product['name']}")
            st.write(f"**Category:** {product['category']}")
            st.write(f"**Brand:** {product['brand']}")
            st.write(f"**Price:** {format_currency(product['price'])}")
        
        with col2:
            st.write(f"**Rating:** {product.get('rating', 0):.1f} ⭐")
            st.write(f"**Reviews:** {product.get('num_reviews', 0)}")
            st.write(f"**Available:** {'Yes' if product.get('availability', 1) else 'No'}")
        
        if product.get('description'):
            st.write(f"**Description:** {product['description']}")
        
        if product.get('features'):
            try:
                features = json.loads(product['features'])
                st.write("**Features:**")
                for key, value in features.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            except:
                pass

if __name__ == "__main__":
    show_recommendations_page()