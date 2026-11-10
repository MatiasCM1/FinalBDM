@echo off
chcp 65001 >nul
title TPF - Instalación de Entorno

echo ======================================================================
echo              TPF - INSTALADOR AUTOMÁTICO DE ENTORNO
echo ======================================================================
echo.
echo Este script configurará todo lo necesario en esta computadora.
echo No necesitás tener VS Code ni conocimientos de programación.
echo.

:: Verificar si Python está instalado
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] No se encontró Python en el sistema.
    echo.
    echo Por favor descargá e instalá Python 3.11 o 3.12 desde:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANTE: Durante la instalación, marcá la casilla que dice:
    echo "[X] Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo [1/3] Python detectado correctamente.
echo [2/3] Instalando dependencias y configurando entorno virtual (.venv)...
echo      (Esto puede demorar unos minutos, no cierres esta ventana)
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Hubo un problema durante la instalación. Revisá los mensajes anteriores.
    pause
    exit /b 1
)

echo.
echo [3/3] Verificando compatibilidad de la GPU y librerías...
call "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\check_env.py"

echo.
echo ======================================================================
echo      ¡INSTALACIÓN COMPLETADA CON ÉXITO!
echo ======================================================================
echo Ya podés cerrar esta ventana y hacer doble clic en:
echo [ 2_CORRER_EXPERIMENTOS.bat ] para iniciar las pruebas cuando quieras.
echo.
pause
