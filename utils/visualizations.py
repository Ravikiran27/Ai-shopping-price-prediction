"""
Visualization utilities for creating interactive charts and graphs
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from config.config import VIZ_CONFIG
from utils.helpers import format_currency, format_number

def create_price_distribution_chart(df: pd.DataFrame, 
                                   price_column: str = 'price',
                                   category_column: str = 'category') -> go.Figure:
    """Create price distribution chart by category"""
    
    fig = px.box(
        df, 
        x=category_column, 
        y=price_column,
        title="Price Distribution by Category",
        color=category_column,
        color_discrete_sequence=VIZ_CONFIG['color_palette']
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Price ($)",
        showlegend=False,
        height=500
    )
    
    fig.update_xaxis(tickangle=45)
    
    return fig

def create_sales_trend_chart(df: pd.DataFrame, 
                            date_column: str = 'timestamp',
                            value_column: str = 'price_paid',
                            title: str = "Sales Trend Over Time") -> go.Figure:
    """Create sales trend chart over time"""
    
    # Aggregate daily sales
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    daily_sales = df_copy.groupby(df_copy[date_column].dt.date)[value_column].agg(['sum', 'count']).reset_index()
    daily_sales.columns = ['Date', 'Revenue', 'Orders']
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add revenue line
    fig.add_trace(
        go.Scatter(
            x=daily_sales['Date'],
            y=daily_sales['Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color=VIZ_CONFIG['color_palette'][0], width=3),
            hovertemplate='Date: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add orders line
    fig.add_trace(
        go.Scatter(
            x=daily_sales['Date'],
            y=daily_sales['Orders'],
            mode='lines+markers',
            name='Orders',
            line=dict(color=VIZ_CONFIG['color_palette'][1], width=3),
            hovertemplate='Date: %{x}<br>Orders: %{y}<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
    
    fig.update_layout(
        title=title,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_category_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create category performance comparison chart"""
    
    # Calculate metrics by category
    category_metrics = df.groupby('category').agg({
        'price_paid': ['sum', 'mean', 'count'],
        'rating': 'mean'
    }).round(2)
    
    category_metrics.columns = ['Total_Revenue', 'Avg_Price', 'Total_Orders', 'Avg_Rating']
    category_metrics = category_metrics.reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Revenue by Category', 'Average Order Value', 
                       'Total Orders', 'Average Rating'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Revenue chart
    fig.add_trace(
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['Total_Revenue'],
            name='Revenue',
            marker_color=VIZ_CONFIG['color_palette'][0],
            hovertemplate='%{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Average price chart
    fig.add_trace(
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['Avg_Price'],
            name='Avg Price',
            marker_color=VIZ_CONFIG['color_palette'][1],
            hovertemplate='%{x}<br>Avg Price: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Orders chart
    fig.add_trace(
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['Total_Orders'],
            name='Orders',
            marker_color=VIZ_CONFIG['color_palette'][2],
            hovertemplate='%{x}<br>Orders: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Rating chart
    fig.add_trace(
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['Avg_Rating'],
            name='Avg Rating',
            marker_color=VIZ_CONFIG['color_palette'][3],
            hovertemplate='%{x}<br>Rating: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Category Performance Dashboard"
    )
    
    return fig

def create_user_behavior_chart(interactions_df: pd.DataFrame) -> go.Figure:
    """Create user behavior analysis chart"""
    
    # Calculate user metrics
    user_metrics = interactions_df.groupby('user_id').agg({
        'product_id': 'count',
        'price_paid': ['sum', 'mean'],
        'interaction_type': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    user_metrics.columns = ['Total_Interactions', 'Total_Spent', 'Avg_Order_Value', 'Interaction_Types']
    user_metrics = user_metrics.reset_index()
    
    # Create scatter plot
    fig = px.scatter(
        user_metrics,
        x='Total_Interactions',
        y='Total_Spent',
        size='Avg_Order_Value',
        color='Avg_Order_Value',
        title='User Behavior Analysis',
        labels={
            'Total_Interactions': 'Number of Interactions',
            'Total_Spent': 'Total Amount Spent ($)',
            'Avg_Order_Value': 'Average Order Value ($)'
        },
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_recommendation_performance_chart(metrics: Dict[str, float]) -> go.Figure:
    """Create recommendation system performance chart"""
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=VIZ_CONFIG['color_palette'][:len(metric_names)],
            text=[f'{v:.3f}' for v in metric_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Recommendation System Performance',
        xaxis_title='Metrics',
        yaxis_title='Score',
        height=400
    )
    
    return fig

def create_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Create feature importance chart for ML models"""
    
    fig = px.bar(
        importance_df.head(15),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Feature Importance',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Create model comparison chart"""
    
    fig = go.Figure()
    
    metrics = ['Test RMSE', 'Test MAE', 'Test R²']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                marker_color=VIZ_CONFIG['color_palette'][i],
                text=comparison_df[metric].round(3),
                textposition='auto'
            ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig

def create_price_prediction_chart(predicted_price: float, 
                                 confidence_interval: Tuple[float, float],
                                 actual_price: Optional[float] = None) -> go.Figure:
    """Create price prediction visualization"""
    
    fig = go.Figure()
    
    # Add predicted price
    fig.add_trace(go.Scatter(
        x=['Predicted'],
        y=[predicted_price],
        mode='markers',
        marker=dict(size=20, color=VIZ_CONFIG['color_palette'][0]),
        name='Predicted Price',
        hovertemplate='Predicted: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=['Predicted', 'Predicted'],
        y=list(confidence_interval),
        mode='lines',
        line=dict(color=VIZ_CONFIG['color_palette'][0], width=2),
        name='Confidence Interval',
        hovertemplate='CI: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add actual price if provided
    if actual_price is not None:
        fig.add_trace(go.Scatter(
            x=['Actual'],
            y=[actual_price],
            mode='markers',
            marker=dict(size=20, color=VIZ_CONFIG['color_palette'][1]),
            name='Actual Price',
            hovertemplate='Actual: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Price Prediction Results',
        yaxis_title='Price ($)',
        height=400,
        showlegend=True
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Create correlation heatmap for numeric features"""
    
    # Select numeric columns
    numeric_cols = df[features].select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_kpi_cards(metrics: Dict[str, Any]) -> None:
    """Create KPI cards using Streamlit columns"""
    
    cols = st.columns(len(metrics))
    
    for i, (metric_name, metric_data) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(metric_data, dict):
                value = metric_data.get('value', 0)
                delta = metric_data.get('delta', None)
                delta_color = metric_data.get('delta_color', 'normal')
            else:
                value = metric_data
                delta = None
                delta_color = 'normal'
            
            # Format value based on type
            if isinstance(value, float):
                if 'price' in metric_name.lower() or 'revenue' in metric_name.lower():
                    formatted_value = format_currency(value)
                elif 'percentage' in metric_name.lower() or 'rate' in metric_name.lower():
                    formatted_value = f"{value:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = format_number(value)
            else:
                formatted_value = str(value)
            
            st.metric(
                label=metric_name,
                value=formatted_value,
                delta=delta,
                delta_color=delta_color
            )

def create_donut_chart(data: pd.Series, title: str = "Distribution") -> go.Figure:
    """Create a donut chart from pandas Series"""
    
    fig = px.pie(
        values=data.values,
        names=data.index,
        title=title,
        hole=0.3,
        color_discrete_sequence=VIZ_CONFIG['color_palette']
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_time_series_chart(df: pd.DataFrame, 
                           date_column: str, 
                           value_column: str,
                           title: str = "Time Series") -> go.Figure:
    """Create time series chart"""
    
    fig = px.line(
        df,
        x=date_column,
        y=value_column,
        title=title,
        markers=True
    )
    
    fig.update_traces(
        line=dict(color=VIZ_CONFIG['color_palette'][0], width=3),
        marker=dict(size=6)
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_gauge_chart(value: float, 
                      min_val: float = 0, 
                      max_val: float = 100,
                      title: str = "Gauge") -> go.Figure:
    """Create a gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': (min_val + max_val) / 2},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': VIZ_CONFIG['color_palette'][0]},
            'steps': [
                {'range': [min_val, max_val * 0.5], 'color': "lightgray"},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig

def create_waterfall_chart(values: List[float], 
                          labels: List[str],
                          title: str = "Waterfall Chart") -> go.Figure:
    """Create a waterfall chart"""
    
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=["relative"] * (len(values) - 1) + ["total"],
        x=labels,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=500
    )
    
    return fig

def style_metric_cards():
    """Apply custom CSS styling to metric cards"""
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] > div {
        width: fit-content;
        margin: auto;
    }
    
    [data-testid="metric-container"] label {
        width: fit-content;
        margin: auto;
        font-weight: bold;
        color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

def create_price_history_chart(price_history_df: pd.DataFrame, 
                              product_name: str = "Product") -> go.Figure:
    """Create price history chart for a product"""
    
    if price_history_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Sort by timestamp
    df = price_history_df.sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Add price history line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines+markers',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add trend line
    if len(df) > 1:
        x_numeric = np.arange(len(df))
        z = np.polyfit(x_numeric, df['price'], 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend:</b> $%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Price History - {product_name}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_price_prediction_with_history_chart(price_history_df: pd.DataFrame,
                                              predictions_df: pd.DataFrame,
                                              product_name: str = "Product") -> go.Figure:
    """Create combined price history and prediction chart"""
    
    fig = go.Figure()
    
    # Historical prices
    if not price_history_df.empty:
        history_df = price_history_df.sort_values('timestamp')
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['price'],
            mode='lines+markers',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6, color='#1f77b4'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
    
    # Future predictions
    if not predictions_df.empty:
        pred_df = predictions_df.copy()
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=6, color='#ff7f0e'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Add confidence bands if available
        if 'confidence' in pred_df.columns:
            upper_bound = pred_df['predicted_price'] * (1 + pred_df['confidence'] * 0.1)
            lower_bound = pred_df['predicted_price'] * (1 - pred_df['confidence'] * 0.1)
            
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=upper_bound,
                mode='lines',
                line=dict(color='rgba(255,127,14,0.2)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=lower_bound,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,127,14,0.2)', width=0),
                name='Confidence Band',
                hoverinfo='skip'
            ))
    
    # Add vertical line to separate history from predictions
    if not price_history_df.empty and not predictions_df.empty:
        last_historical_date = pd.to_datetime(price_history_df['timestamp']).max()
        fig.add_vline(
            x=last_historical_date,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prediction Start",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"Price History & Predictions - {product_name}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

def create_price_trends_chart(trends_df: pd.DataFrame) -> go.Figure:
    """Create price trends chart by category"""
    
    if trends_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price trends data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    
    fig = go.Figure()
    
    # Add line for each category
    categories = trends_df['category'].unique()
    colors = px.colors.qualitative.Set1
    
    for i, category in enumerate(categories):
        category_data = trends_df[trends_df['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['date'],
            y=category_data['avg_price'],
            mode='lines+markers',
            name=category,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{category}</b><br><b>Date:</b> %{{x}}<br><b>Avg Price:</b> $%{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Price Trends by Category",
        xaxis_title="Date",
        yaxis_title="Average Price ($)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_price_volatility_chart(price_history_df: pd.DataFrame) -> go.Figure:
    """Create price volatility analysis chart"""
    
    if price_history_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price data available for volatility analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Calculate volatility metrics by product
    volatility_data = []
    
    for product_id in price_history_df['product_id'].unique():
        product_data = price_history_df[price_history_df['product_id'] == product_id].sort_values('timestamp')
        
        if len(product_data) > 1:
            prices = product_data['price']
            returns = prices.pct_change().dropna()
            
            volatility_data.append({
                'product_id': product_id,
                'product_name': product_data['product_name'].iloc[0],
                'category': product_data['category'].iloc[0],
                'volatility': returns.std() * 100,  # Convert to percentage
                'price_range': prices.max() - prices.min(),
                'avg_price': prices.mean()
            })
    
    if not volatility_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for volatility analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    vol_df = pd.DataFrame(volatility_data)
    
    # Create scatter plot
    fig = px.scatter(
        vol_df,
        x='avg_price',
        y='volatility',
        color='category',
        size='price_range',
        hover_data=['product_name'],
        title="Price Volatility Analysis",
        labels={
            'avg_price': 'Average Price ($)',
            'volatility': 'Price Volatility (%)',
            'price_range': 'Price Range ($)'
        }
    )
    
    fig.update_layout(height=500)
    
    return fig