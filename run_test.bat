@echo off
set VENV_PYTHON=c:\Users\ACER\Desktop\Game\ML2\.venv\Scripts\python.exe

echo Running diagnostic test...
%VENV_PYTHON% test_script.py

echo.
echo Running AL pipeline with random strategy...
%VENV_PYTHON% pipeline_AL.py --methods random --budgets 10 20

pause
