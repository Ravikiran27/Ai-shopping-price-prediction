#!/bin/bash

echo "========================================"
echo "    🛒 SmartCommerce-AI Setup"
echo "========================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python --version

echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🗄️ Initializing database..."
python -c "
import sys
sys.path.append('.')
from data.database import init_database
init_database()
print('✅ Database initialized successfully!')
"

echo ""
echo "🎯 Setup complete!"
echo ""
echo "Choose your interface:"
echo "  👤 User Interface:     ./start_user_app.sh"
echo "  🔧 Advanced Interface: ./start_advanced.sh"
echo ""
echo "Or run manually:"
echo "  streamlit run user_app.py --server.port=8502"
echo "  streamlit run app.py --server.port=8501"
echo ""
echo "========================================"