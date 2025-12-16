@echo off
echo Select action:
echo 1. Activate the environment and launch Streamlit
echo 2. Activate the environment and run Jupyter
echo 3. Install/update libraries
echo 4. Check installation
echo 5. Log out

set /p choice="Enter number: "

if "%choice%"=="1" (
    call venv\Scripts\activate
    streamlit run app.py
) else if "%choice%"=="2" (
    call venv\Scripts\activate
    jupyter notebook
) else if "%choice%"=="3" (
    call venv\Scripts\activate
    pip install -r requirements.txt
) else if "%choice%"=="4" (
    call venv\Scripts\activate
    python -c "import pandas; import streamlit; import tensorflow; print('Все ОК!')"
) else (
    echo Goodbye!
)