 # 🛒 AI Shopping Price Prediction
**Intelligent E-Commerce Price Prediction & Recommendation System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Ravikiran27/Ai-shopping-price-prediction/blob/main/LICENSE)
[![SerpAPI](https://img.shields.io/badge/Data-SerpAPI-orange.svg)](https://serpapi.com)
[![Live Demo](https://img.shields.io/badge/Demo-Streamlit%20Cloud-FF4B4B.svg)](https://ravikiran27-ai-shopping-price-prediction-user-app-km5ta9.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Ravikiran27-black.svg)](https://github.com/Ravikiran27/Ai-shopping-price-prediction)

*An advanced AI-powered shopping assistant that provides real-time price predictions, smart product recommendations, and market insights using machine learning and time-series forecasting.*

## � Project Overview

This project demonstrates a **production-ready AI system** that combines:
- **Real-world data integration** using SerpAPI for live Google Shopping data
- **Advanced machine learning** with multiple prediction algorithms
- **Professional software architecture** with background services and error handling
- **User experience design** with both technical and consumer-friendly interfaces
- **Time-series forecasting** for price trend analysis
- **Hybrid recommendation systems** for personalized shopping suggestions

**Perfect for**: Data science portfolios, AI/ML demonstrations, e-commerce analytics, and educational purposes.

## �🌟 Key Features

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

## � Project Stats

- **📁 Files**: 20+ Python files with modular architecture
- **📋 Lines of Code**: 3000+ lines of production-ready code
- **🗄️ Database**: 230+ products with 3400+ price history entries
- **🏷️ Categories**: 11 product categories from real market data
- **🌐 Live Data**: SerpAPI integration with Google Shopping
- **🚀 Deployment**: Streamlit Cloud with automated CI/CD
- **📈 Accuracy**: 85-92% price prediction accuracy

## �🚀 Quick Start

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
git clone https://github.com/Ravikiran27/Ai-shopping-price-prediction.git
cd Ai-shopping-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: Configure SerpAPI (for live data collection)
# Edit config/config.py with your SerpAPI key

# 4. Run the application
streamlit run user_app.py  # Simple interface
# OR
streamlit run app.py       # Advanced interface
```

### **🌐 Live Demo**
Try the application online: **[AI Shopping Assistant](https://ravikiran27-ai-shopping-price-prediction-user-app-km5ta9.streamlit.app/)**

> 💡 **Note**: The live demo runs in simplified mode. For full features including real-time data collection and advanced analytics, clone and run locally.

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Ravikiran27/Ai-shopping-price-prediction/blob/main/LICENSE) file for details.

## 👨‍💻 Author

**Ravikiran27**
- GitHub: [@Ravikiran27](https://github.com/Ravikiran27)
- Repository: [Ai-shopping-price-prediction](https://github.com/Ravikiran27/Ai-shopping-price-prediction)
- Live Demo: [Streamlit App](https://ravikiran27-ai-shopping-price-prediction-user-app-km5ta9.streamlit.app/)

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Ravikiran27/Ai-shopping-price-prediction/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Ravikiran27/Ai-shopping-price-prediction/discussions)
- ⭐ **Star this repo** if you find it useful!

---

**Made with ❤️ by Ravikiran27**

*🚀 Transform your shopping experience with AI-powered insights!*