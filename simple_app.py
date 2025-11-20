"""
Simple user-focused Streamlit application for E-Commerce Price Prediction System
This version hides all technical complexity and provides a clean user experience
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

# Import the simple user interface
from pages.user_interface import show_simple_user_interface
from services.background_service import background_service

def initialize_app():
    """Initialize the Streamlit application"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Smart Shopping Assistant",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UX
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stTextInput > div > div {
        border-radius: 10px;
    }
    
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Hide technical elements */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    .stAlert > div {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Setup logging (minimal for end users)
    setup_logging(
        log_level="WARNING",  # Only show warnings and errors to users
        log_file=None
    )
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        logging.info(f"App initialized with session ID: {st.session_state.session_id}")

def main():
    """Main application function"""
    
    # Initialize the app
    initialize_app()
    
    # Show loading screen for first-time users
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        
        # Welcome screen
        st.title("🛒 Welcome to Smart Shopping Assistant!")
        st.markdown("""
        ### Your AI-Powered Shopping Companion
        
        🎯 **Smart Features:**
        - 🔍 **Product Search**: Find the best products across categories
        - 📈 **Price Tracking**: Monitor price changes and get alerts
        - 🎯 **Personalized Recommendations**: Discover products you'll love
        - 💡 **Market Insights**: Stay informed about price trends
        
        Everything works automatically in the background - just start shopping!
        """)
        
        if st.button("🚀 Start Shopping", type="primary"):
            st.session_state.first_load = False
            st.rerun()
    
    else:
        # Show the main user interface
        try:
            show_simple_user_interface()
        except Exception as e:
            st.error("Something went wrong. Please refresh the page.")
            logging.error(f"Error in main interface: {str(e)}")
            
            # Show a retry button
            if st.button("🔄 Refresh"):
                st.rerun()

if __name__ == "__main__":
    main()