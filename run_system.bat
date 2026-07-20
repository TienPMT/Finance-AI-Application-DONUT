@echo off
title Smart Expense Tracker and Insight Launcher
cls

echo =======================================================================
echo         SMART EXPENSE TRACKER AND INSIGHT UNIFIED RUNNER
echo =======================================================================
echo.

:: Step 1: Start Deep Learning FastAPI
echo [1/3] Starting Deep Learning API (Python FastAPI)...
cd /d "%~dp0ai-service"
start "Deep Learning API" cmd /k "set "PYTHONIOENCODING=utf-8" && .venv\Scripts\activate.bat && uvicorn ai_service:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for Deep Learning API to start on port 8000...
powershell -Command "while ($true) { try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 8000); if ($c.Connected) { $c.Close(); break; } } catch {} Start-Sleep -Seconds 2 }"
echo.
echo [OK] Deep Learning API is ready!
echo -----------------------------------------------------------------------

:: Step 2: Start Java Spring Boot Backend
echo [2/3] Starting Backend (Java Spring Boot)...
cd /d "%~dp0core-backend"
start "Spring Boot Backend" cmd /k "set "JAVA_HOME=D:\Sorfware\jdk-21" && mvnw.cmd spring-boot:run"

echo Waiting for Backend to start on port 8081 (this might take up to a minute for compilation and startup)...
powershell -Command "while ($true) { try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 8081); if ($c.Connected) { $c.Close(); break; } } catch {} Start-Sleep -Seconds 2 }"
echo.
echo [OK] Backend Spring Boot is ready!
echo -----------------------------------------------------------------------

:: Step 3: Run Flutter Mobile App
echo [3/3] Starting Mobile App (Flutter)...
cd /d "%~dp0LTDD_Finance"
echo.
echo Please choose your target device for the Flutter app:
echo [1] Web Browser (Chrome) - localhost
echo [2] Android Emulator - using 10.0.2.2
echo [3] Real Device / Virtual Machine - using PC IP: 192.168.100.4
echo.
set /p choice="Enter your choice (1, 2, or 3) [Default is 1]: "

if "%choice%"=="2" (
    echo Launching on Android Emulator...
    call flutter run -d emulator --dart-define=API_BASE=http://10.0.2.2:8081 --dart-define=ML_BASE=http://10.0.2.2:8000
) else if "%choice%"=="3" (
    echo Launching on Real Device / VM - SM G991N...
    echo Setting up ADB port forwarding over USB...
    adb reverse tcp:8081 tcp:8081
    adb reverse tcp:8000 tcp:8000
    call flutter run -d R3CR30P2LZV --dart-define=API_BASE=http://127.0.0.1:8081 --dart-define=ML_BASE=http://127.0.0.1:8000
) else (
    echo Launching on Chrome browser...
    call flutter run -d chrome
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Flutter run failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo =======================================================================
echo          ALL SERVICES STARTED SUCCESSFULLY!
echo =======================================================================
pause
