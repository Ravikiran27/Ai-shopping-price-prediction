"""
Analytics dashboard page for the e-commerce system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

from data.database import DatabaseManager
from utils.visualizations import (
    create_sales_trend_chart, create_category_performance_chart,
    create_user_behavior_chart, create_donut_chart, create_kpi_cards,
    style_metric_cards
)
from utils.helpers import format_currency, format_number, calculate_growth_rate

def show_analytics_page():
    """Display the analytics dashboard"""
    
    st.title("📊 Analytics Dashboard")
    st.markdown("Comprehensive insights into your e-commerce performance")
    
    # Apply custom styling
    style_metric_cards()
    
    # Initialize database connection
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Sidebar filters
    st.sidebar.header("📅 Filters")
    
    # Date range selector
    date_range = st.sidebar.selectbox(
        "Select Time Period",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Last year", "All time"]
    )
    
    # Category filter
    categories = ["All"] + list(db_manager.get_products()['category'].unique())
    selected_category = st.sidebar.selectbox("Category", categories)
    
    # Get analytics data
    days_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last year": 365,
        "All time": None
    }
    
    days = days_map[date_range]
    analytics_data = db_manager.get_sales_analytics(days) if days else db_manager.get_sales_analytics(365)
    
    # Get additional data
    products_df = db_manager.get_products()
    
    # Filter by category if selected
    if selected_category != "All":
        products_df = products_df[products_df['category'] == selected_category]
    
    # Calculate KPIs
    total_products = len(products_df)
    avg_price = products_df['price'].mean()
    total_users = len(db_manager.get_users())
    
    # Get interaction data for more metrics
    interactions_query = """
        SELECT i.*, p.name as product_name, p.category, p.price
        FROM interactions i
        JOIN products p ON i.product_id = p.id
        WHERE i.interaction_type = 'purchase'
    """
    
    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        interactions_query += f" AND i.timestamp >= '{cutoff_date}'"
    
    with db_manager.get_connection() as conn:
        interactions_df = pd.read_sql_query(interactions_query, conn)
    
    # Filter interactions by category
    if selected_category != "All":
        interactions_df = interactions_df[interactions_df['category'] == selected_category]
    
    # Calculate revenue metrics
    total_revenue = interactions_df['price_paid'].sum() if not interactions_df.empty else 0
    total_orders = len(interactions_df)
    avg_order_value = interactions_df['price_paid'].mean() if not interactions_df.empty else 0
    
    # Calculate growth rates (mock comparison with previous period)
    prev_revenue = total_revenue * 0.85  # Simulated previous period data
    revenue_growth = calculate_growth_rate(total_revenue, prev_revenue)
    
    # Display KPIs
    st.subheader("📈 Key Performance Indicators")
    
    kpi_metrics = {
        "Total Revenue": {
            "value": total_revenue,
            "delta": f"{revenue_growth:+.1f}%",
            "delta_color": "normal" if revenue_growth > 0 else "inverse"
        },
        "Total Orders": {
            "value": total_orders,
            "delta": "+12%",  # Mock data
            "delta_color": "normal"
        },
        "Average Order Value": {
            "value": avg_order_value,
            "delta": "-2.3%",  # Mock data
            "delta_color": "inverse"
        },
        "Total Products": {
            "value": total_products,
            "delta": "+5",  # Mock data
            "delta_color": "normal"
        },
        "Active Users": {
            "value": total_users,
            "delta": "+8.2%",  # Mock data
            "delta_color": "normal"
        }
    }
    
    create_kpi_cards(kpi_metrics)
    
    st.divider()
    
    # Main dashboard content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Sales Trend")
        if not interactions_df.empty:
            sales_chart = create_sales_trend_chart(interactions_df)
            st.plotly_chart(sales_chart, use_container_width=True)
        else:
            st.info("No sales data available for the selected period")
    
    with col2:
        st.subheader("🏷️ Category Distribution")
        if not products_df.empty:
            category_dist = products_df['category'].value_counts()
            donut_chart = create_donut_chart(category_dist, "Products by Category")
            st.plotly_chart(donut_chart, use_container_width=True)
        else:
            st.info("No product data available")
    
    st.divider()
    
    # Category performance
    st.subheader("📊 Category Performance Analysis")
    if not interactions_df.empty:
        category_chart = create_category_performance_chart(interactions_df)
        st.plotly_chart(category_chart, use_container_width=True)
    else:
        st.info("No interaction data available for category analysis")
    
    st.divider()
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛍️ Top Products")
        if not interactions_df.empty:
            top_products = (interactions_df.groupby(['product_name', 'category'])
                          .agg({
                              'price_paid': 'sum',
                              'quantity': 'sum'
                          })
                          .sort_values('price_paid', ascending=False)
                          .head(10)
                          .reset_index())
            
            if not top_products.empty:
                for idx, row in top_products.iterrows():
                    st.write(f"**{row['product_name']}**")
                    st.write(f"Category: {row['category']} | Revenue: {format_currency(row['price_paid'])} | Sold: {row['quantity']}")
                    st.write("---")
            else:
                st.info("No product sales data available")
        else:
            st.info("No sales data available")
    
    with col2:
        st.subheader("👥 User Behavior")
        if not interactions_df.empty:
            user_behavior_chart = create_user_behavior_chart(interactions_df)
            st.plotly_chart(user_behavior_chart, use_container_width=True)
        else:
            st.info("No user interaction data available")
    
    st.divider()
    
    # Price analysis
    st.subheader("💲 Price Analysis")
    
    if not products_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution by category
            fig_price_dist = px.box(
                products_df,
                x='category',
                y='price',
                title='Price Distribution by Category',
                color='category'
            )
            fig_price_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_price_dist, use_container_width=True)
        
        with col2:
            # Rating vs Price scatter
            fig_rating_price = px.scatter(
                products_df,
                x='price',
                y='rating',
                size='num_reviews',
                color='category',
                title='Rating vs Price Analysis',
                hover_data=['name']
            )
            st.plotly_chart(fig_rating_price, use_container_width=True)
    
    st.divider()
    
    # Data export section
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Products Data"):
            csv = products_df.to_csv(index=False)
            st.download_button(
                label="Download Products CSV",
                data=csv,
                file_name=f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Sales Data"):
            if not interactions_df.empty:
                csv = interactions_df.to_csv(index=False)
                st.download_button(
                    label="Download Sales CSV",
                    data=csv,
                    file_name=f"sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No sales data to export")
    
    with col3:
        if st.button("Export Analytics Report"):
            # Create summary report
            report_data = {
                'metric': list(kpi_metrics.keys()),
                'value': [kpi_metrics[k]['value'] for k in kpi_metrics.keys()],
                'change': [kpi_metrics[k]['delta'] for k in kpi_metrics.keys()]
            }
            report_df = pd.DataFrame(report_data)
            
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="Download Report CSV",
                data=csv,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Real-time updates section
    with st.expander("🔄 Real-time Updates"):
        st.info("Analytics data is updated in real-time as new transactions occur.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Data"):
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)")
            if auto_refresh:
                st.rerun()

if __name__ == "__main__":
    show_analytics_page()