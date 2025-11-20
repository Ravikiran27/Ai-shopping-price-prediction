@echo off
echo ========================================
echo   SmartCommerce-AI - Advanced Interface
echo ========================================
echo Starting developer/analyst interface...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
streamlit run app.py --server.port=8501
pause