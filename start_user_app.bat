@echo off
echo Starting Smart Shopping Assistant...
echo.
echo Please wait while the system initializes...
echo This may take a few moments on first startup.
echo.

REM Install required packages if needed
pip install -r requirements.txt

REM Start the user-friendly application
streamlit run user_app.py --server.port 8501 --server.address localhost

pause