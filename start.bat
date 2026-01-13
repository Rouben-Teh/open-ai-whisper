@echo off
REM 切换到当前 bat 所在目录
cd /d %~dp0

call venv\Scripts\activate.bat

python app.py

pause