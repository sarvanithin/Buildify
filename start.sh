#!/bin/bash
# Drafted — start backend + frontend
cd "$(dirname "$0")"

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Starting Ollama..."
  ollama serve &
  sleep 3
fi

# Start backend
echo "Starting backend on :8000..."
cd backend && python3 main.py &
BACKEND_PID=$!
cd ..

# Start frontend
echo "Starting frontend on :5173..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "  App: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
