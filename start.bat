@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Product Research Agent - Startup
echo ============================================

REM --- Load .env if present ---
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%B"=="" (
                set "%%A=%%B"
            )
        )
    )
    echo [OK] Loaded .env
)

REM --- Start CLIProxyAPI if setup-token mode ---
if "%LLM_MODE%"=="setup-token" (
    echo.
    echo [1/2] Starting CLIProxyAPI proxy on localhost:8317 ...
    if not exist "C:\cliproxyapi\cli-proxy-api.exe" (
        echo [ERROR] cli-proxy-api.exe not found at C:\cliproxyapi\cli-proxy-api.exe
        echo         Install CLIProxyAPI first.
        exit /b 1
    )
    start "CLIProxyAPI" cmd /c "C:\cliproxyapi\cli-proxy-api.exe --config C:\cliproxyapi\config.yaml 2>&1"
    timeout /t 2 /nobreak >nul
    echo [OK] CLIProxyAPI proxy started
)

REM --- Run the research agent ---
echo.
echo [2/2] Starting Product Research Agent ...
echo.

conda run -n research_tools python -m src %*

endlocal
