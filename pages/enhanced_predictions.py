"""
Enhanced price prediction page with historical data and time-series forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from data.database import DatabaseManager
from models.price_predictor import PricePredictionModel
from models.time_series_predictor import TimeSeriesPricePredictor
from utils.visualizations import (
    create_price_prediction_chart, create_feature_importance_chart,
    create_model_comparison_chart, create_price_history_chart,
    create_price_prediction_with_history_chart, create_price_trends_chart,
    create_price_volatility_chart
)
from utils.helpers import format_currency

def show_enhanced_price_prediction_page():
    """Display the enhanced price prediction interface with historical data"""
    
    st.title("🔮 Advanced Price Prediction & Forecasting")
    st.markdown("Predict product prices using historical data and AI-powered forecasting models")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'price_model' not in st.session_state:
        st.session_state.price_model = PricePredictionModel()
    
    if 'ts_model' not in st.session_state:
        st.session_state.ts_model = TimeSeriesPricePredictor()
    
    db_manager = st.session_state.db_manager
    price_model = st.session_state.price_model
    ts_model = st.session_state.ts_model
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price History & Forecasting", 
        "🎯 Single Prediction", 
        "📊 Market Trends", 
        "💹 Price Volatility",
        "🤖 Model Training",
        "⚙️ Price Management"
    ])
    
    with tab1:
        show_price_history_forecasting(db_manager, ts_model)
    
    with tab2:
        show_single_prediction_interface(price_model, db_manager)
    
    with tab3:
        show_market_trends_interface(db_manager)
    
    with tab4:
        show_price_volatility_interface(db_manager)
    
    with tab5:
        show_model_training_interface(price_model, ts_model, db_manager)
    
    with tab6:
        show_price_management_interface(db_manager)

def show_price_history_forecasting(db_manager: DatabaseManager, ts_model: TimeSeriesPricePredictor):
    """Show price history and forecasting interface"""
    
    st.subheader("📈 Price History & Future Forecasting")
    
    # Get products for selection
    products_df = db_manager.get_products()
    
    if products_df.empty:
        st.warning("No products found. Please add some products first.")
        return
    
    # Product selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        product_names = products_df['name'].tolist()
        selected_product_name = st.selectbox("Select Product", product_names)
        selected_product = products_df[products_df['name'] == selected_product_name].iloc[0]
    
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 90, 30)
    
    # Get price history for selected product
    price_history = db_manager.get_price_history(selected_product['id'])
    
    if price_history.empty:
        st.info("No price history found for this product. Let's create some historical data.")
        
        # Generate sample price history
        if st.button("Generate Sample Price History"):
            generate_sample_price_history(db_manager, selected_product['id'], selected_product['price'])
            st.rerun()
    else:
        # Display current price info
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = price_history.iloc[0]['price']  # Most recent price
        oldest_price = price_history.iloc[-1]['price']   # Oldest price
        price_change = current_price - oldest_price
        price_change_pct = (price_change / oldest_price) * 100 if oldest_price != 0 else 0
        
        with col1:
            st.metric("Current Price", format_currency(current_price))
        with col2:
            st.metric("Price Change", 
                     format_currency(price_change), 
                     delta=f"{price_change_pct:.1f}%")
        with col3:
            st.metric("Historical Range", 
                     f"{format_currency(price_history['price'].min())} - {format_currency(price_history['price'].max())}")
        with col4:
            st.metric("Data Points", len(price_history))
        
        # Price history chart
        st.subheader(f"Price History - {selected_product['name']}")
        history_chart = create_price_history_chart(price_history, selected_product['name'])
        st.plotly_chart(history_chart, use_container_width=True)
        
        # Time-series forecasting
        st.subheader("🔮 Future Price Forecasting")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner("Training time-series model and generating forecast..."):
                    try:
                        # Train time-series model if not already trained or if we have new data
                        all_price_history = db_manager.get_price_history()
                        all_products = db_manager.get_products()
                        
                        if len(all_price_history) >= 50:  # Minimum data requirement
                            metrics = ts_model.train(all_price_history, all_products)
                            st.success(f"Model trained! RMSE: {metrics['rmse']:.2f}")
                            
                            # Generate predictions
                            product_info = {
                                'price': selected_product['price'],
                                'rating': selected_product['rating'],
                                'num_reviews': selected_product['num_reviews']
                            }
                            
                            predictions = ts_model.predict_future_prices(
                                selected_product['id'], 
                                forecast_days, 
                                price_history,
                                product_info
                            )
                            
                            # Store predictions in session state
                            st.session_state.current_predictions = predictions
                            st.session_state.current_product_history = price_history
                            st.session_state.current_product_name = selected_product['name']
                            
                        else:
                            st.error("Insufficient historical data for time-series forecasting. Need at least 50 price points across all products.")
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
        
        # Display predictions if available
        if 'current_predictions' in st.session_state:
            predictions = st.session_state.current_predictions
            
            # Combined chart with history and predictions
            combined_chart = create_price_prediction_with_history_chart(
                st.session_state.current_product_history,
                predictions,
                st.session_state.current_product_name
            )
            st.plotly_chart(combined_chart, use_container_width=True)
            
            # Prediction summary
            st.subheader("📊 Forecast Summary")
            
            avg_predicted_price = predictions['predicted_price'].mean()
            min_predicted_price = predictions['predicted_price'].min()
            max_predicted_price = predictions['predicted_price'].max()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Forecast Price", format_currency(avg_predicted_price))
            with col2:
                forecast_change = avg_predicted_price - current_price
                forecast_change_pct = (forecast_change / current_price) * 100
                st.metric("Expected Change", 
                         format_currency(forecast_change),
                         delta=f"{forecast_change_pct:.1f}%")
            with col3:
                st.metric("Forecast Range", 
                         f"{format_currency(min_predicted_price)} - {format_currency(max_predicted_price)}")
            with col4:
                volatility = (max_predicted_price - min_predicted_price) / avg_predicted_price * 100
                st.metric("Forecast Volatility", f"{volatility:.1f}%")

def show_single_prediction_interface(price_model: PricePredictionModel, db_manager: DatabaseManager):
    """Show single product price prediction interface"""
    
    st.subheader("🎯 Single Product Price Prediction")
    st.info("This uses traditional ML models based on product features (not time-series)")
    
    # Get sample data for dropdowns
    products_df = db_manager.get_products()
    categories = sorted(products_df['category'].unique()) if not products_df.empty else []
    brands = sorted(products_df['brand'].dropna().unique()) if not products_df.empty else []
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input("Product Name", value="Premium Smartphone Pro Max")
        category = st.selectbox("Category", categories if categories else ["Electronics"])
        brand = st.selectbox("Brand", brands if brands else ["Apple"])
        
        rating = st.slider("Rating", 1.0, 5.0, 4.2, 0.1)
        num_reviews = st.number_input("Number of Reviews", min_value=0, value=150, step=1)
    
    with col2:
        description = st.text_area(
            "Product Description", 
            value="Latest premium smartphone with advanced camera, fast processor, and premium build quality."
        )
        
        # Additional features
        st.subheader("Additional Features")
        storage = st.selectbox("Storage", ["64GB", "128GB", "256GB", "512GB", "1TB"])
        color = st.selectbox("Color", ["Black", "White", "Silver", "Gold", "Blue"])
        warranty = st.selectbox("Warranty", ["1 year", "2 years", "3 years"])
    
    # Model selection
    available_models = ['random_forest', 'gradient_boosting', 'linear_regression']
    selected_model = st.selectbox("Select Model", available_models)
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary"):
        try:
            # Prepare product data
            product_data = {
                'name': product_name,
                'category': category,
                'brand': brand,
                'description': description,
                'rating': rating,
                'num_reviews': num_reviews,
                'features': {
                    'storage': storage,
                    'color': color,
                    'warranty': warranty,
                    'weight': '0.5 kg',
                    'in_stock': True,
                    'free_shipping': True
                }
            }
            
            # Make prediction
            predicted_price = price_model.predict_single_product(product_data, selected_model)
            
            # Display result
            st.success(f"Predicted Price: {format_currency(predicted_price)}")
            
            # Find similar products for comparison
            similar_products = products_df[
                (products_df['category'] == category) & 
                (products_df['brand'] == brand)
            ].head(5)
            
            if not similar_products.empty:
                st.subheader("Similar Products Comparison")
                comparison_df = similar_products[['name', 'price', 'rating', 'num_reviews']].copy()
                comparison_df['predicted_price'] = predicted_price
                st.dataframe(comparison_df)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please train the models first in the Model Training tab.")

def show_market_trends_interface(db_manager: DatabaseManager):
    """Show market trends analysis"""
    
    st.subheader("📊 Market Price Trends")
    
    # Time period selection
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2)
    
    with col2:
        if st.button("Refresh Trends"):
            st.rerun()
    
    # Get price trends
    trends_df = db_manager.get_price_trends(days_back)
    
    if trends_df.empty:
        st.info("No price trends data available. Price changes will appear here once products have price history.")
    else:
        # Trends chart
        trends_chart = create_price_trends_chart(trends_df)
        st.plotly_chart(trends_chart, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Trend Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Categories Tracked", trends_df['category'].nunique())
        
        with col2:
            total_changes = trends_df['price_changes'].sum()
            st.metric("Total Price Changes", total_changes)
        
        with col3:
            avg_price = trends_df['avg_price'].mean()
            st.metric("Average Market Price", format_currency(avg_price))
        
        # Detailed trends table
        st.subheader("📋 Detailed Trends")
        st.dataframe(trends_df)

def show_price_volatility_interface(db_manager: DatabaseManager):
    """Show price volatility analysis"""
    
    st.subheader("💹 Price Volatility Analysis")
    
    # Get all price history
    price_history = db_manager.get_price_history()
    
    if price_history.empty:
        st.info("No price history available for volatility analysis.")
    else:
        # Volatility chart
        volatility_chart = create_price_volatility_chart(price_history)
        st.plotly_chart(volatility_chart, use_container_width=True)
        
        st.markdown("""
        **How to read this chart:**
        - **X-axis**: Average price of products
        - **Y-axis**: Price volatility (standard deviation of price changes)
        - **Size**: Price range (difference between highest and lowest price)
        - **Color**: Product category
        
        Products in the upper right are high-priced and highly volatile.
        Products in the lower left are low-priced and stable.
        """)

def show_model_training_interface(price_model: PricePredictionModel, ts_model: TimeSeriesPricePredictor, db_manager: DatabaseManager):
    """Show model training interface"""
    
    st.subheader("🤖 Model Training & Management")
    
    # Traditional ML Models
    st.subheader("Traditional ML Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Traditional Models"):
            train_traditional_models(price_model, db_manager)
    
    with col2:
        if st.button("Load Saved Models"):
            try:
                price_model.load_models()
                st.success("Traditional models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
    
    # Time-Series Model
    st.subheader("Time-Series Forecasting Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Time-Series Model"):
            train_time_series_model(ts_model, db_manager)
    
    with col2:
        if st.button("Save Time-Series Model"):
            try:
                model_path = "models/saved_models/time_series_model.joblib"
                ts_model.save_model(model_path)
                st.success("Time-series model saved successfully!")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

def show_price_management_interface(db_manager: DatabaseManager):
    """Show price management interface"""
    
    st.subheader("⚙️ Price Management")
    
    # Get products
    products_df = db_manager.get_products()
    
    if products_df.empty:
        st.warning("No products found.")
        return
    
    # Product selection for price update
    selected_product_name = st.selectbox("Select Product to Update Price", products_df['name'].tolist())
    selected_product = products_df[products_df['name'] == selected_product_name].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Current Price: {format_currency(selected_product['price'])}")
        new_price = st.number_input("New Price", min_value=0.01, value=float(selected_product['price']), step=0.01)
    
    with col2:
        st.write("") # Spacing
        if st.button("Update Price"):
            if db_manager.update_product_price(selected_product['id'], new_price):
                st.success(f"Price updated successfully! New price: {format_currency(new_price)}")
                st.rerun()
            else:
                st.error("Failed to update price.")
    
    # Bulk price generation
    st.subheader("📊 Generate Historical Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_to_generate = st.slider("Days of History", 7, 365, 30)
    
    with col2:
        if st.button("Generate Price History for All Products"):
            generate_bulk_price_history(db_manager, days_to_generate)

def train_traditional_models(price_model: PricePredictionModel, db_manager: DatabaseManager):
    """Train traditional ML models"""
    with st.spinner("Training traditional ML models..."):
        try:
            products_df = db_manager.get_products()
            
            if len(products_df) < 50:
                st.error("Need at least 50 products to train models. Please add more sample data.")
                return
            
            X, y = price_model.prepare_data(products_df)
            results = price_model.train_models(X, y)
            price_model.save_models()
            
            st.success("Traditional models trained successfully!")
            
            for model_name, metrics in results.items():
                if 'error' not in metrics:
                    st.write(f"**{model_name.title()}**")
                    st.write(f"- RMSE: {metrics['test_rmse']:.2f}")
                    st.write(f"- R²: {metrics['test_r2']:.3f}")
        
        except Exception as e:
            st.error(f"Error training models: {str(e)}")

def train_time_series_model(ts_model: TimeSeriesPricePredictor, db_manager: DatabaseManager):
    """Train time-series forecasting model"""
    with st.spinner("Training time-series model..."):
        try:
            all_price_history = db_manager.get_price_history()
            all_products = db_manager.get_products()
            
            if len(all_price_history) < 50:
                st.error("Need at least 50 historical price points to train time-series model.")
                return
            
            metrics = ts_model.train(all_price_history, all_products)
            
            st.success("Time-series model trained successfully!")
            st.write(f"**Performance Metrics:**")
            st.write(f"- RMSE: {metrics['rmse']:.2f}")
            st.write(f"- MAE: {metrics['mae']:.2f}")
            st.write(f"- MAPE: {metrics['mape']:.2f}%")
        
        except Exception as e:
            st.error(f"Error training time-series model: {str(e)}")

def generate_sample_price_history(db_manager: DatabaseManager, product_id: int, base_price: float, days: int = 30):
    """Generate sample price history for a product"""
    import random
    from datetime import timedelta
    
    current_price = base_price
    
    for i in range(days):
        # Add some randomness to price changes
        change_pct = random.uniform(-0.05, 0.05)  # -5% to +5% daily change
        current_price *= (1 + change_pct)
        
        # Ensure price doesn't go below 10% of original or above 200%
        current_price = max(base_price * 0.1, min(base_price * 2.0, current_price))
        
        timestamp = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d %H:%M:%S')
        db_manager.add_price_history(product_id, current_price, timestamp)

def generate_bulk_price_history(db_manager: DatabaseManager, days: int):
    """Generate price history for all products"""
    with st.spinner(f"Generating {days} days of price history..."):
        products_df = db_manager.get_products()
        
        for _, product in products_df.iterrows():
            generate_sample_price_history(db_manager, product['id'], product['price'], days)
        
        st.success(f"Generated {days} days of price history for {len(products_df)} products!")
        st.rerun()

# Main function for the page
def show_price_prediction_page():
    """Main function to display the enhanced price prediction page"""
    show_enhanced_price_prediction_page()