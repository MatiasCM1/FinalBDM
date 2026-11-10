@echo off
chcp 65001 >nul
title TPF - Runner de Experimentos TabPFN (5 Nodos)

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ======================================================================
    echo [ALERTA] Todavía no se configuró el entorno en esta máquina.
    echo ======================================================================
    echo Por favor ejecutá primero el archivo:
    echo [ 1_INSTALAR_TODO.bat ] haciendo doble clic sobre él.
    echo.
    pause
    exit /b 1
)

:MENU
cls
echo ======================================================================
echo           TPF - PANEL DE EXPERIMENTOS TABPFN (FLOTA 5 NODOS)
echo ======================================================================
echo  Podés pausar en cualquier momento con [Ctrl + C] y reanudar después.
echo  Las corridas ya terminadas NUNCA se repiten.
echo ======================================================================
echo.
echo  Seleccioná el rol de esta computadora:
echo.
echo   [1] PC 1 (RTX 3070)      --^> IEEE-CIS (Semillas 0 y 1)           [~19 hs]
echo   [2] PC 2 (1660 Super #1) --^> Home Credit (Semillas 0 a 7)         [~21 hs]
echo   [3] PC 3 (1660 Super #2) --^> ULB (Semillas 0 a 7)                 [~7 hs]
echo.
echo   (Nota: Las Semillas 2 a 7 de IEEE-CIS van en las 2 cuentas de Colab Pro)
echo.
echo   [4] Ver Estado           --^> Ver cuántas corridas faltan y cuántas terminaron
echo   [5] Personalizado        --^> Elegir dataset y semilla manualmente
echo   [0] Salir
echo.
echo ======================================================================
set /p OPCION="Ingresá una opción [0-5] y presioná Enter: "

if "%OPCION%"=="1" goto CORRER_PC1
if "%OPCION%"=="2" goto CORRER_PC2
if "%OPCION%"=="3" goto CORRER_PC3
if "%OPCION%"=="4" goto VER_ESTADO
if "%OPCION%"=="5" goto PERSONALIZADO
if "%OPCION%"=="0" exit /b 0

echo Opción inválida.
timeout /t 2 >nul
goto MENU

:CORRER_PC1
cls
echo ======================================================================
echo [INICIANDO] PC 1 (RTX 3070) - IEEE-CIS Semillas 0 y 1
echo ======================================================================
echo Tip: Cuando quieras jugar o usar la PC, apretá [Ctrl + C] para parar.
echo      Cuando vuelvas a abrir este archivo, continúa donde quedó.
echo.
"%PYTHON%" "%~dp0scripts\run_grid.py" --datasets ieee_cis --seeds 0 1 --n-estimators 4
pause
goto MENU

:CORRER_PC2
cls
echo ======================================================================
echo [INICIANDO] PC 2 (1660 Super #1) - Home Credit Semillas 0 a 7
echo ======================================================================
echo Tip: Cuando quieras jugar o usar la PC, apretá [Ctrl + C] para parar.
echo      Cuando vuelvas a abrir este archivo, continúa donde quedó.
echo.
"%PYTHON%" "%~dp0scripts\run_grid.py" --datasets home_credit --seeds 0 1 2 3 4 5 6 7 --n-estimators 4
pause
goto MENU

:CORRER_PC3
cls
echo ======================================================================
echo [INICIANDO] PC 3 (1660 Super #2) - ULB Semillas 0 a 7
echo ======================================================================
echo Tip: Cuando quieras jugar o usar la PC, apretá [Ctrl + C] para parar.
echo      Cuando vuelvas a abrir este archivo, continúa donde quedó.
echo.
"%PYTHON%" "%~dp0scripts\run_grid.py" --datasets ulb --seeds 0 1 2 3 4 5 6 7 --n-estimators 4
pause
goto MENU

:VER_ESTADO
cls
echo Consultando estado de avance de las 360 combinaciones...
echo.
"%PYTHON%" "%~dp0scripts\run_grid.py" --status-only
echo.
pause
goto MENU

:PERSONALIZADO
cls
echo ======================================================================
echo                       EJECUCIÓN PERSONALIZADA
echo ======================================================================
echo Datasets disponibles: ulb, home_credit, ieee_cis
set /p DATASET_ELEGIDO="Ingresá el dataset (o varios separados por espacio): "
set /p SEMILLAS_ELEGIDAS="Ingresá las semillas (ej. 0 o 0 1 2): "

"%PYTHON%" "%~dp0scripts\run_grid.py" --datasets %DATASET_ELEGIDO% --seeds %SEMILLAS_ELEGIDAS% --n-estimators 4
pause
goto MENU
