"""
Price prediction page for the e-commerce system
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from data.database import DatabaseManager
from models.price_predictor import PricePredictionModel
from utils.visualizations import (
    create_price_prediction_chart, create_feature_importance_chart,
    create_model_comparison_chart
)
from utils.helpers import format_currency

def show_price_prediction_page():
    """Display the price prediction interface"""
    
    st.title("🔮 Price Prediction")
    st.markdown("Predict product prices using AI-powered machine learning models")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'price_model' not in st.session_state:
        st.session_state.price_model = PricePredictionModel()
    
    db_manager = st.session_state.db_manager
    price_model = st.session_state.price_model
    
    # Sidebar for model management
    st.sidebar.header("🤖 Model Management")
    
    # Model training section
    with st.sidebar.expander("Train Models"):
        if st.button("Train All Models"):
            with st.spinner("Training models..."):
                try:
                    # Get training data
                    products_df = db_manager.get_products()
                    
                    if len(products_df) < 50:
                        st.error("Need at least 50 products to train models. Please add more sample data.")
                    else:
                        # Prepare data
                        X, y = price_model.prepare_data(products_df)
                        
                        # Train models
                        results = price_model.train_models(X, y)
                        
                        # Save models
                        price_model.save_models()
                        
                        st.success("Models trained successfully!")
                        
                        # Display results
                        for model_name, metrics in results.items():
                            if 'error' not in metrics:
                                st.write(f"**{model_name.title()}**")
                                st.write(f"- Test RMSE: {metrics['test_rmse']:.2f}")
                                st.write(f"- Test R²: {metrics['test_r2']:.3f}")
                
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
    
    # Load existing models
    if st.sidebar.button("Load Saved Models"):
        try:
            price_model.load_models()
            st.sidebar.success("Models loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading models: {str(e)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict Price", "📊 Model Performance", "🔍 Feature Analysis", "📈 Batch Prediction"])
    
    with tab1:
        show_single_prediction_interface(price_model, db_manager)
    
    with tab2:
        show_model_performance_interface(price_model)
    
    with tab3:
        show_feature_analysis_interface(price_model)
    
    with tab4:
        show_batch_prediction_interface(price_model, db_manager)

def show_single_prediction_interface(price_model, db_manager):
    """Show single product price prediction interface"""
    
    st.subheader("Predict Price for a Single Product")
    
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
                    'weight': '0.5 kg',  # Default values
                    'in_stock': True,
                    'free_shipping': True
                }
            }
            
            # Make prediction
            with st.spinner("Predicting price..."):
                prediction_result = price_model.predict_single_product(
                    product_data, 
                    model_name=selected_model
                )
            
            # Display results
            st.success("Price prediction completed!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Predicted Price",
                    value=format_currency(prediction_result['predicted_price']),
                    delta=None
                )
                
                st.metric(
                    label="Model Used",
                    value=prediction_result['model_used'].replace('_', ' ').title(),
                    delta=None
                )
                
                # Confidence interval
                ci_lower, ci_upper = prediction_result['confidence_interval']
                st.write(f"**Confidence Interval:**")
                st.write(f"{format_currency(ci_lower)} - {format_currency(ci_upper)}")
            
            with col2:
                # Visualization
                prediction_chart = create_price_prediction_chart(
                    prediction_result['predicted_price'],
                    prediction_result['confidence_interval']
                )
                st.plotly_chart(prediction_chart, use_container_width=True)
            
            # Feature contributions
            if prediction_result['feature_contributions']:
                st.subheader("Feature Contributions")
                contributions_df = pd.DataFrame(
                    list(prediction_result['feature_contributions'].items()),
                    columns=['Feature', 'Contribution']
                ).sort_values('Contribution', key=abs, ascending=False).head(10)
                
                st.dataframe(contributions_df, use_container_width=True)
            
            # Model performance info
            if prediction_result['model_performance']:
                with st.expander("Model Performance Details"):
                    performance = prediction_result['model_performance']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'test_rmse' in performance:
                            st.metric("Test RMSE", f"{performance['test_rmse']:.2f}")
                    
                    with col2:
                        if 'test_r2' in performance:
                            st.metric("Test R²", f"{performance['test_r2']:.3f}")
                    
                    with col3:
                        if 'test_mae' in performance:
                            st.metric("Test MAE", f"{performance['test_mae']:.2f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Sample predictions section
    st.divider()
    st.subheader("📋 Quick Predictions")
    st.markdown("Try these sample products for quick testing:")
    
    sample_products = [
        {
            "name": "Gaming Laptop Ultra",
            "category": "Electronics",
            "brand": "Dell",
            "description": "High-performance gaming laptop with RTX graphics",
            "features": {"storage": "1TB", "color": "Black", "warranty": "2 years"}
        },
        {
            "name": "Designer Jeans",
            "category": "Clothing",
            "brand": "Nike",
            "description": "Premium denim jeans with modern fit",
            "features": {"size": "M", "color": "Blue", "material": "Denim"}
        },
        {
            "name": "Smart Home Speaker",
            "category": "Electronics",
            "brand": "Amazon",
            "description": "Voice-controlled smart speaker with WiFi",
            "features": {"color": "White", "wireless": "WiFi", "warranty": "1 year"}
        }
    ]
    
    cols = st.columns(len(sample_products))
    
    for i, sample in enumerate(sample_products):
        with cols[i]:
            st.write(f"**{sample['name']}**")
            st.write(f"Category: {sample['category']}")
            st.write(f"Brand: {sample['brand']}")
            
            if st.button(f"Predict {sample['name']}", key=f"sample_{i}"):
                try:
                    sample['rating'] = 4.0
                    sample['num_reviews'] = 100
                    
                    prediction_result = price_model.predict_single_product(sample)
                    st.success(f"Predicted: {format_currency(prediction_result['predicted_price'])}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def show_model_performance_interface(price_model):
    """Show model performance comparison"""
    
    st.subheader("Model Performance Comparison")
    
    if price_model.model_performance:
        # Create comparison DataFrame
        comparison_df = price_model.get_model_comparison()
        
        if not comparison_df.empty:
            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            comparison_chart = create_model_comparison_chart(comparison_df)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Best model highlight
            best_model_idx = comparison_df['Test RMSE'].idxmin()
            best_model = comparison_df.loc[best_model_idx, 'Model']
            best_rmse = comparison_df.loc[best_model_idx, 'Test RMSE']
            best_r2 = comparison_df.loc[best_model_idx, 'Test R²']
            
            st.success(f"🏆 **Best Model**: {best_model.title()} (RMSE: {best_rmse:.2f}, R²: {best_r2:.3f})")
        else:
            st.info("No model performance data available. Please train models first.")
    else:
        st.info("No model performance data available. Please train models first.")
    
    # Cross-validation results
    if price_model.model_performance:
        st.subheader("Cross-Validation Results")
        
        cv_data = []
        for model_name, metrics in price_model.model_performance.items():
            if 'error' not in metrics and 'cv_rmse_mean' in metrics:
                cv_data.append({
                    'Model': model_name,
                    'CV RMSE Mean': metrics['cv_rmse_mean'],
                    'CV RMSE Std': metrics['cv_rmse_std']
                })
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            st.dataframe(cv_df, use_container_width=True)

def show_feature_analysis_interface(price_model):
    """Show feature importance analysis"""
    
    st.subheader("Feature Importance Analysis")
    
    if price_model.feature_importance:
        # Model selection for feature importance
        available_models = list(price_model.feature_importance.keys())
        selected_model = st.selectbox("Select Model for Feature Analysis", available_models)
        
        # Get feature importance
        importance_df = price_model.get_feature_importance(selected_model, top_n=20)
        
        if not importance_df.empty:
            # Display chart
            importance_chart = create_feature_importance_chart(importance_df)
            st.plotly_chart(importance_chart, use_container_width=True)
            
            # Display table
            st.subheader("Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
            
            # Insights
            st.subheader("Key Insights")
            top_feature = importance_df.iloc[0]['Feature']
            top_importance = importance_df.iloc[0]['Importance']
            
            st.write(f"• **Most Important Feature**: {top_feature} ({top_importance:.3f})")
            st.write(f"• **Top 5 Features** account for {importance_df.head(5)['Importance'].sum():.1%} of model decisions")
            
            # Feature categories
            text_features = importance_df[importance_df['Feature'].str.contains('length|word|sentiment', case=False)]
            if not text_features.empty:
                st.write(f"• **Text Features Impact**: {len(text_features)} features related to product descriptions")
            
            numeric_features = importance_df[importance_df['Feature'].str.contains('price|rating|reviews', case=False)]
            if not numeric_features.empty:
                st.write(f"• **Numeric Features Impact**: {len(numeric_features)} numerical features are important")
        
        else:
            st.info("No feature importance data available for the selected model.")
    else:
        st.info("No feature importance data available. Please train models first.")

def show_batch_prediction_interface(price_model, db_manager):
    """Show batch prediction interface"""
    
    st.subheader("Batch Price Prediction")
    st.markdown("Upload a CSV file or select products from database for batch prediction")
    
    # Method selection
    method = st.radio("Select Method", ["Upload CSV", "Select from Database"])
    
    if method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Predict Prices for All Products"):
                    with st.spinner("Making predictions..."):
                        predictions = []
                        
                        for idx, row in df.iterrows():
                            try:
                                product_data = row.to_dict()
                                # Ensure required fields
                                if 'features' not in product_data:
                                    product_data['features'] = {}
                                
                                result = price_model.predict_single_product(product_data)
                                predictions.append({
                                    'Index': idx,
                                    'Product Name': product_data.get('name', 'Unknown'),
                                    'Predicted Price': result['predicted_price'],
                                    'Model Used': result['model_used']
                                })
                            except Exception as e:
                                st.warning(f"Error predicting for row {idx}: {str(e)}")
                        
                        if predictions:
                            predictions_df = pd.DataFrame(predictions)
                            st.success(f"Predictions completed for {len(predictions)} products!")
                            st.dataframe(predictions_df, use_container_width=True)
                            
                            # Download results
                            csv = predictions_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name=f"price_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Select from Database
        products_df = db_manager.get_products()
        
        if not products_df.empty:
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                categories = ["All"] + sorted(products_df['category'].unique())
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                max_products = st.number_input("Max Products to Predict", min_value=1, max_value=100, value=10)
            
            # Filter products
            filtered_df = products_df.copy()
            if selected_category != "All":
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            
            filtered_df = filtered_df.head(max_products)
            
            st.write(f"Selected {len(filtered_df)} products for prediction:")
            st.dataframe(filtered_df[['name', 'category', 'brand', 'price']], use_container_width=True)
            
            if st.button("Predict Prices for Selected Products"):
                with st.spinner("Making predictions..."):
                    predictions = []
                    
                    for idx, row in filtered_df.iterrows():
                        try:
                            product_data = {
                                'name': row['name'],
                                'category': row['category'],
                                'brand': row['brand'],
                                'description': row.get('description', ''),
                                'rating': row.get('rating', 4.0),
                                'num_reviews': row.get('num_reviews', 0),
                                'features': json.loads(row.get('features', '{}')) if row.get('features') else {}
                            }
                            
                            result = price_model.predict_single_product(product_data)
                            
                            predictions.append({
                                'Product ID': row['id'],
                                'Product Name': row['name'],
                                'Actual Price': row['price'],
                                'Predicted Price': result['predicted_price'],
                                'Difference': result['predicted_price'] - row['price'],
                                'Error %': abs(result['predicted_price'] - row['price']) / row['price'] * 100,
                                'Model Used': result['model_used']
                            })
                        
                        except Exception as e:
                            st.warning(f"Error predicting for product {row['name']}: {str(e)}")
                    
                    if predictions:
                        predictions_df = pd.DataFrame(predictions)
                        
                        # Calculate overall metrics
                        mean_error = predictions_df['Error %'].mean()
                        
                        st.success(f"Predictions completed! Average error: {mean_error:.1f}%")
                        st.dataframe(predictions_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Error %", f"{mean_error:.1f}%")
                        
                        with col2:
                            st.metric("Best Prediction", f"{predictions_df['Error %'].min():.1f}%")
                        
                        with col3:
                            st.metric("Worst Prediction", f"{predictions_df['Error %'].max():.1f}%")
                        
                        # Download results
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="Download Batch Predictions CSV",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        else:
            st.info("No products available in database. Please add some products first.")

if __name__ == "__main__":
    show_price_prediction_page()