@echo off
REM === Navigate to your project folder ===
cd /d C:\Code\stock_dashboard

REM === Activate your virtual environment ===
call venv\Scripts\activate.bat

REM === Run Streamlit dashboard ===
streamlit run app.py

REM === Keep window open after exit ===
pause
