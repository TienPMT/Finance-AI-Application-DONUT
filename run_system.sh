#!/bin/bash

# Exit immediately if any command fails
# set -e

echo "======================================================================="
echo "        SMART EXPENSE TRACKER & INSIGHT UNIFIED RUNNER"
echo "======================================================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Step 1: Start Deep Learning FastAPI
echo "[1/3] Khởi động Deep Learning API (Python FastAPI)..."
cd "$SCRIPT_DIR/ai-service"
source .venv/bin/activate
PYTHONIOENCODING=utf-8 uvicorn ai_service:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &
API_PID=$!

echo "Đang chờ Deep Learning API hoạt động trên cổng 8000..."
while ! curl -s http://localhost:8000/docs >/dev/null; do
    sleep 2
done
echo "[OK] Deep Learning API đã sẵn sàng! (PID: $API_PID)"
echo "-----------------------------------------------------------------------"

# Step 2: Start Java Spring Boot Backend
echo "[2/3] Khởi động Backend (Java Spring Boot)..."
cd "$SCRIPT_DIR/core-backend"
# Set JAVA_HOME to standard if available or keep default
# export JAVA_HOME=/path/to/jdk-21
./mvnw spring-boot:run > springboot.log 2>&1 &
BACKEND_PID=$!

echo "Đang chờ Backend hoạt động trên cổng 8081..."
while ! curl -s http://localhost:8081 >/dev/null; do
    sleep 2
done
echo "[OK] Backend Spring Boot đã sẵn sàng! (PID: $BACKEND_PID)"
echo "-----------------------------------------------------------------------"

# Step 3: Run Flutter Mobile App
echo "[3/3] Khởi động Mobile App (Flutter)..."
cd "$SCRIPT_DIR/LTDD_Finance"
echo ""
echo "Please choose your target device for the Flutter app:"
echo "[1] Web Browser (Chrome) - localhost"
echo "[2] Android Emulator - using 10.0.2.2"
echo "[3] Real Device / Virtual Machine - using PC IP: 192.168.100.4"
echo ""
read -p "Enter your choice (1, 2, or 3) [Default is 1]: " choice

if [ "$choice" = "2" ]; then
    echo "Launching on Android Emulator..."
    flutter run -d android --dart-define=API_BASE=http://10.0.2.2:8081 --dart-define=ML_BASE=http://10.0.2.2:8000
elif [ "$choice" = "3" ]; then
    echo "Launching on Real Device / VM..."
    flutter run -d android --dart-define=API_BASE=http://192.168.100.4:8081 --dart-define=ML_BASE=http://192.168.100.4:8000
else
    echo "Launching on Chrome browser..."
    flutter run -d chrome
fi

# Stop background services when Flutter exits
cleanup() {
    echo ""
    echo "Dừng các dịch vụ chạy ngầm..."
    kill $API_PID $BACKEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT
