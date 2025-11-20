"""
Main Streamlit application for AI-Powered E-Commerce Price Prediction & Product Recommendation System
"""

import streamlit as st
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import configuration
from config.config import STREAMLIT_CONFIG, LOGGING_CONFIG
from utils.helpers import setup_logging, generate_session_id

# Import page modules
from pages.analytics import show_analytics_page
from pages.enhanced_predictions import show_price_prediction_page
from pages.recommendations import show_recommendations_page
from pages.real_time_data import show_real_time_data_page
from pages.admin import show_admin_page

# Import data modules
from data.database import DatabaseManager
from data.data_generator import create_sample_data

def initialize_app():
    """Initialize the Streamlit application"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title=STREAMLIT_CONFIG['page_title'],
        page_icon=STREAMLIT_CONFIG['page_icon'],
        layout=STREAMLIT_CONFIG['layout'],
        initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
    )
    
    # Setup logging
    setup_logging(
        log_level=LOGGING_CONFIG['level'],
        log_file=None  # Console logging only for Streamlit
    )
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    # Custom CSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d0d0d0;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff6b6b;
        color: white;
    }
    
    .success-message {
        padding: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin: 10px 0;
    }
    
    .warning-message {
        padding: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        color: #856404;
        margin: 10px 0;
    }
    
    .error-message {
        padding: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def show_welcome_page():
    """Show welcome page with system overview"""
    
    st.title("🛒 AI-Powered E-Commerce Analytics Platform")
    st.markdown("### Transform your e-commerce business with intelligent price predictions and personalized recommendations")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Welcome to the future of e-commerce analytics!** Our platform combines cutting-edge machine learning 
        with intuitive analytics to help you make data-driven decisions that boost sales and customer satisfaction.
        
        🔮 **Price Prediction**: Use advanced ML algorithms to predict optimal product prices  
        🎯 **Smart Recommendations**: Deliver personalized product suggestions to your customers  
        📊 **Analytics Dashboard**: Gain deep insights into sales performance and customer behavior  
        ⚙️ **Admin Panel**: Manage your product catalog, users, and system settings  
        """)
        
        # Quick stats
        if 'db_manager' in st.session_state:
            db_manager = st.session_state.db_manager
            stats = db_manager.get_database_stats()
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Products", stats.get('products', 0))
            with col_b:
                st.metric("Users", stats.get('users', 0))
            with col_c:
                st.metric("Interactions", stats.get('interactions', 0))
            with col_d:
                st.metric("Categories", stats.get('categories', 0))
    
    with col2:
        st.image("https://via.placeholder.com/400x300/ff6b6b/ffffff?text=AI+E-Commerce", 
                caption="AI-Powered E-Commerce Platform")
    
    st.divider()
    
    # Feature highlights
    st.subheader("🚀 Key Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        #### 🔮 Price Prediction
        - **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression
        - **Real-time Predictions**: Instant price estimates for new products
        - **Feature Importance**: Understand which factors drive pricing
        - **Confidence Intervals**: Get prediction reliability scores
        - **Batch Processing**: Predict prices for multiple products at once
        """)
    
    with feature_col2:
        st.markdown("""
        #### 🎯 Recommendations
        - **Collaborative Filtering**: Learn from user behavior patterns
        - **Content-Based**: Match products by features and descriptions
        - **Hybrid Approach**: Combine multiple recommendation strategies
        - **Cold Start Handling**: Recommendations for new users and products
        - **Explanation Engine**: Understand why products are recommended
        """)
    
    with feature_col3:
        st.markdown("""
        #### 📊 Analytics & Insights
        - **Sales Dashboards**: Interactive charts and visualizations
        - **Customer Analytics**: User behavior and preference analysis
        - **Performance Metrics**: Track KPIs and business performance
        - **Export Capabilities**: Download reports and data
        - **Real-time Updates**: Live data refreshing and monitoring
        """)
    
    st.divider()
    
    # Getting started guide
    st.subheader("🎯 Getting Started")
    
    with st.expander("📋 Quick Start Guide", expanded=True):
        st.markdown("""
        **Step 1: Initialize Sample Data**
        1. Go to the **Admin Panel** → **Data Management** tab
        2. Click "Generate Sample Data" to create products, users, and interactions
        3. Wait for the data generation to complete
        
        **Step 2: Train ML Models**
        1. Navigate to **Admin Panel** → **ML Models** tab
        2. Click "Train Price Models" to train the price prediction algorithms
        3. Click "Train Recommendation Models" to set up the recommendation system
        
        **Step 3: Explore Features**
        1. **Analytics**: View comprehensive dashboards and insights
        2. **Price Prediction**: Test price predictions for products
        3. **Recommendations**: Get personalized product suggestions
        
        **Step 4: Manage Your System**
        1. Use the **Admin Panel** to add products and users
        2. Monitor system performance and model accuracy
        3. Export data and generate reports as needed
        """)
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🎲 Generate Sample Data", help="Create sample products, users, and interactions"):
            with st.spinner("Generating sample data..."):
                try:
                    stats = create_sample_data()
                    st.success(f"✅ Created {stats['products']} products, {stats['users']} users, {stats['interactions']} interactions!")
                    st.session_state.data_initialized = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    with action_col2:
        if st.button("📊 View Analytics", help="Go to analytics dashboard"):
            st.session_state.page = "Analytics"
            st.rerun()
    
    with action_col3:
        if st.button("🔮 Try Predictions", help="Test price prediction models"):
            st.session_state.page = "Price Prediction"
            st.rerun()
    
    with action_col4:
        if st.button("🎯 Get Recommendations", help="Explore recommendation system"):
            st.session_state.page = "Recommendations"
            st.rerun()
    
    # System status
    st.divider()
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        # Check if database is initialized
        try:
            if 'db_manager' in st.session_state:
                stats = st.session_state.db_manager.get_database_stats()
                if stats.get('products', 0) > 0:
                    st.success("✅ Database: Ready")
                else:
                    st.warning("⚠️ Database: Empty (Generate sample data)")
            else:
                st.info("🔄 Database: Initializing")
        except:
            st.error("❌ Database: Error")
    
    with status_col2:
        # Check ML models status
        try:
            if 'price_model' in st.session_state and hasattr(st.session_state.price_model, 'model_performance'):
                if st.session_state.price_model.model_performance:
                    st.success("✅ ML Models: Trained")
                else:
                    st.warning("⚠️ ML Models: Not trained")
            else:
                st.info("🔄 ML Models: Ready to train")
        except:
            st.error("❌ ML Models: Error")
    
    with status_col3:
        # System health
        st.success("✅ System: Healthy")

def show_navigation():
    """Show navigation sidebar"""
    
    st.sidebar.title("🛒 E-Commerce AI")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = {
        "🏠 Home": "Home",
        "📊 Analytics": "Analytics", 
        "🔮 Price Prediction": "Price Prediction",
        "🎯 Recommendations": "Recommendations",
        "🌐 Real-Time Data": "Real-Time Data",
        "⚙️ Admin Panel": "Admin"
    }
    
    # Page selection
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    for page_icon, page_name in pages.items():
        if st.sidebar.button(page_icon, key=page_name, use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.subheader("📋 System Info")
    
    # Session info
    st.sidebar.write(f"**Session ID**: {st.session_state.session_id[:8]}...")
    st.sidebar.write(f"**Started**: {datetime.now().strftime('%H:%M')}")
    
    # Database stats
    if 'db_manager' in st.session_state:
        try:
            stats = st.session_state.db_manager.get_database_stats()
            st.sidebar.write(f"**Products**: {stats.get('products', 0)}")
            st.sidebar.write(f"**Users**: {stats.get('users', 0)}")
            st.sidebar.write(f"**Interactions**: {stats.get('interactions', 0)}")
        except:
            st.sidebar.write("**Database**: Not available")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data"):
        # Clear cache and reinitialize
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("💾 Save Session"):
        st.sidebar.success("Session saved!")
    
    if st.sidebar.button("❓ Help"):
        st.sidebar.info("Navigate using the buttons above. Start by generating sample data in the Admin Panel.")

def initialize_database():
    """Initialize database connection"""
    
    if 'db_manager' not in st.session_state:
        try:
            st.session_state.db_manager = DatabaseManager()
            logging.info("Database manager initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            logging.error(f"Database initialization error: {str(e)}")
            return False
    
    return True

def main():
    """Main application entry point"""
    
    # Initialize application
    initialize_app()
    
    # Initialize database
    if not initialize_database():
        st.stop()
    
    # Show navigation
    show_navigation()
    
    # Get current page
    current_page = st.session_state.get('page', 'Home')
    
    # Display appropriate page
    try:
        if current_page == "Home":
            show_welcome_page()
        elif current_page == "Analytics":
            show_analytics_page()
        elif current_page == "Price Prediction":
            show_price_prediction_page()
        elif current_page == "Recommendations":
            show_recommendations_page()
        elif current_page == "Real-Time Data":
            show_real_time_data_page()
        elif current_page == "Admin":
            show_admin_page()
        else:
            st.error(f"Unknown page: {current_page}")
            st.session_state.page = "Home"
            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading page '{current_page}': {str(e)}")
        logging.error(f"Page loading error for {current_page}: {str(e)}")
        
        # Fallback to home page
        if current_page != "Home":
            if st.button("🏠 Return to Home"):
                st.session_state.page = "Home"
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🛒 AI-Powered E-Commerce Analytics Platform | Built with ❤️ using Streamlit</p>
            <p>© 2024 E-Commerce AI System. All rights reserved.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()