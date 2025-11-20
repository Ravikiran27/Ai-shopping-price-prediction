@echo off
echo Starting Smart Shopping Assistant...
echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Launching application...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run simple_app.py