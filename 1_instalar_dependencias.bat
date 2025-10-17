@echo off
echo ========================================
echo Instalando dependencias para o executavel.
echo ========================================
echo.

pip install pillow numpy matplotlib --upgrade
pip install pyinstaller --upgrade

echo.
echo ========================================
echo Instalacao concluida!
echo ========================================
pause
