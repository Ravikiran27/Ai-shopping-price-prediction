 # 🛒 SmartCommerce-AI
**Intelligent E-Commerce Price Prediction & Recommendation System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SerpAPI](https://img.shields.io/badge/Data-SerpAPI-orange.svg)](https://serpapi.com)

*An advanced AI-powered shopping assistant that provides real-time price predictions, smart product recommendations, and market insights using machine learning and time-series forecasting.*

## 🌟 Key Features

### 🔮 **AI-Powered Price Prediction**
- **Historical Price Analysis**: Tracks real price changes over time (3400+ price points)
- **Time-Series Forecasting**: Predicts future prices using advanced ML algorithms
- **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression
- **Confidence Intervals**: Visual prediction uncertainty with confidence bands

### 🛍️ **Smart Recommendations**
- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based recommendations
- **Personalized Discovery**: Learns from user behavior patterns
- **Similar Product Matching**: Advanced feature-based product similarity

### 🌐 **Real-Time Data Collection**
- **SerpAPI Integration**: Live product data from Google Shopping
- **Automated Updates**: Background services update prices every 30 minutes
- **230+ Products**: Across 11 categories with real market data

### 📊 **Advanced Analytics**
- **Interactive Dashboards**: Beautiful visualizations with Plotly
- **Market Intelligence**: Price volatility and trend analysis
- **Business Insights**: Category performance and sales analytics

### 🎯 **Dual Interface Design**
- **Simple Mode**: Clean, user-friendly interface for consumers
- **Advanced Mode**: Technical interface for developers and analysts
- **Mobile Responsive**: Optimized for all devices

## 🚀 Quick Start

### **Simple Interface (Recommended for End Users)**
```bash
# Windows
start_user_app.bat

# Linux/Mac
chmod +x start_user_app.sh && ./start_user_app.sh
```
**→ Opens at: http://localhost:8502**

### **Advanced Interface (For Developers)**
```bash
# Windows  
start.bat

# Linux/Mac
chmod +x start.sh && ./start.sh
```
**→ Opens at: http://localhost:8501**

## 📋 Installation

### **Prerequisites**
- Python 3.8 or higher
- 4GB RAM (8GB recommended)
- Internet connection for real-time data

### **Step-by-Step Setup**
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SmartCommerce-AI.git
cd SmartCommerce-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: Configure SerpAPI (for live data collection)
# Edit config/config.py with your SerpAPI key

# 4. Run the application
streamlit run user_app.py  # Simple interface
# OR
streamlit run app.py       # Advanced interface
```

## 🏗️ Architecture Overview

```
shopping-price-prediction/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config/
│   └── config.py              # Configuration settings
├── data/
│   ├── __init__.py
│   ├── database.py            # Database operations
│   ├── data_generator.py      # Sample data generation
│   └── preprocessor.py        # Data preprocessing
├── models/
│   ├── __init__.py
│   ├── price_predictor.py     # Price prediction models
│   └── recommender.py         # Recommendation system
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Helper functions
│   └── visualizations.py     # Chart and graph utilities
└── pages/
    ├── __init__.py
    ├── analytics.py           # Analytics dashboard
    ├── predictions.py         # Price prediction interface
    ├── recommendations.py     # Recommendation interface
    └── admin.py              # Admin panel
```

## Usage

### Price Prediction
1. Navigate to the "Price Prediction" page
2. Enter product details (name, category, brand, features)
3. Click "Predict Price" to get ML-powered price estimates
4. View confidence intervals and feature importance

### Product Recommendations
1. Go to the "Recommendations" page
2. Select a user or enter preferences
3. Choose recommendation type (collaborative/content-based)
4. View personalized product suggestions

### Analytics Dashboard
1. Access the "Analytics" page
2. Explore sales trends, revenue metrics
3. Analyze product performance and market insights
4. Export reports for further analysis

### Admin Panel
1. Visit the "Admin" page
2. Manage product catalog and user data
3. Retrain ML models with updated data
4. Configure system settings

## Machine Learning Models

### Price Prediction
- **Random Forest Regressor**: Ensemble method for robust predictions
- **Gradient Boosting**: Sequential learning for improved accuracy
- **Linear Regression**: Baseline model for comparison
- **Feature Engineering**: Categorical encoding, scaling, and transformation

### Recommendation System
- **Collaborative Filtering**: Matrix factorization and similarity-based
- **Content-Based**: TF-IDF and cosine similarity on product features
- **Hybrid Model**: Weighted combination of multiple approaches
- **Cold Start Handling**: Popularity-based recommendations for new users

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with SQLite database
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy, TextBlob

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue on GitHub or contact the development team.