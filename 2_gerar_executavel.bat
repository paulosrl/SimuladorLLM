@echo off
echo ========================================
echo Gerando executavel do Simulador LLM
echo MPPA - CIAA
echo ========================================
echo.

echo Removendo builds anteriores...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "simulador_llm.spec" del "simulador_llm.spec"

echo.
echo Gerando executavel (isso pode demorar alguns minutos)...
echo.

pyinstaller --name="simulador_llm_Windows" ^
    --onefile ^
    --windowed ^
    --icon=Mui.png ^
    --add-data="Mui.png;." ^
    --hidden-import=PIL ^
    --hidden-import=PIL._imagingtk ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=numpy ^
    --hidden-import=matplotlib ^
    --hidden-import=mpl_toolkits.mplot3d ^
    --collect-all matplotlib ^
    --collect-all tkinter ^
    simulador_llm_Windows.py

echo.
echo ========================================
if exist "dist\simulador_llm_Windows.exe" (
    echo SUCESSO! Executavel gerado em: dist\simulador_llm_Windows.exe
    echo.
    echo Copiando logo para a pasta dist...
    copy Mui.png dist\Mui.png
    echo.
    echo Pronto para distribuir!
) else (
    echo ERRO: Nao foi possivel gerar o executavel.
    echo Verifique os erros acima.
)
echo ========================================
echo.
pause
