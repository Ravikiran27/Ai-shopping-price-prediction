"""
User-friendly Streamlit application for e-commerce price prediction
All technical complexity is hidden - users just interact with the interface
"""

import streamlit as st
import sys
from pathlib import Path
import logging
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import configuration
from config.config import STREAMLIT_CONFIG
from utils.helpers import setup_logging, generate_session_id

# Import user interface
from pages.user_interface import show_simple_user_interface

# Import services
from services.simple_background_service import start_background_service

def initialize_user_app():
    """Initialize the user-friendly Streamlit application"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="🛒 Smart Shopping Assistant",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup minimal logging
    setup_logging(log_level="WARNING", log_file=None)
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    
    if 'app_initialized' not in st.session_state:
        with st.spinner("🚀 Starting Smart Shopping Assistant..."):
            # Start background services
            start_background_service()
            time.sleep(3)  # Give background services time to initialize
            st.session_state.app_initialized = True

def main():
    """Main application function"""
    try:
        # Initialize the application
        initialize_user_app()
        
        # Show the user interface
        show_simple_user_interface()
        
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        st.info("Please refresh the page to restart the application.")
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()