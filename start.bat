@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================
echo   Product Research Agent
echo ============================================

REM Load .env if present. Keep values simple: KEY=VALUE, no shell expansion.
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "key=%%A"
        set "value=%%B"
        if not "!key!"=="" if not "!key:~0,1!"=="#" (
            set "!key!=!value!"
        )
    )
    echo [OK] Loaded .env
)

if "%LLM_MODE%"=="" set "LLM_MODE=setup-token"
if "%LLM_PROXY_URL%"=="" set "LLM_PROXY_URL=http://localhost:8317"
if "%CLIPROXYAPI_EXE%"=="" set "CLIPROXYAPI_EXE=C:\cliproxyapi\cli-proxy-api.exe"
if "%CLIPROXYAPI_CONFIG%"=="" set "CLIPROXYAPI_CONFIG=C:\cliproxyapi\config.yaml"
set "NEEDS_LLM=0"
if /I "%~1"=="research" set "NEEDS_LLM=1"

REM Start CLIProxyAPI only when setup-token mode is requested and the proxy is not reachable.
if "%NEEDS_LLM%"=="1" if /I "%LLM_MODE%"=="setup-token" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='%LLM_PROXY_URL%'.TrimEnd('/') + '/v1/models'; try { Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
    if errorlevel 1 (
        echo [INFO] Starting CLIProxyAPI at %LLM_PROXY_URL%
        if not exist "%CLIPROXYAPI_EXE%" (
            echo [ERROR] CLIProxyAPI executable not found: %CLIPROXYAPI_EXE%
            echo         Set CLIPROXYAPI_EXE in .env or install CLIProxyAPI.
            exit /b 1
        )
        if exist "%CLIPROXYAPI_CONFIG%" (
            start "CLIProxyAPI" cmd /c ""%CLIPROXYAPI_EXE%" --config "%CLIPROXYAPI_CONFIG%""
        ) else (
            start "CLIProxyAPI" cmd /c ""%CLIPROXYAPI_EXE%""
        )
        timeout /t 2 /nobreak >nul
    ) else (
        echo [OK] CLIProxyAPI is already reachable
    )
)

echo.
echo [INFO] Running Product Research Agent
echo.

if /I "%CONDA_DEFAULT_ENV%"=="research_tools" (
    python -m src %*
) else (
    where conda >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] conda not found. Activate research_tools and run: python -m src %*
        exit /b 1
    )
    conda run -n research_tools python -m src %*
)

endlocal
