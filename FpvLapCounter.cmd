@echo off
set "APPDIR=%~dp0"
set "APPDIR=%APPDIR:~0,-1%"
wt -w 0 nt --title "FPV Lap Counter" -d "%APPDIR%" C:\Python310\python.exe -m streamlit run app.py
