#!/bin/bash

# Navigate to the repository directory
REPO_DIR=$(dirname "$(realpath "$0")")
cd "$REPO_DIR" || { echo "Failed to navigate to the repository directory"; exit 1; }

# Configure git pull strategy
git config pull.rebase false

# Pull the latest changes
echo "Pulling latest changes from the repository..."
git reset --hard  # Ensure no local changes cause conflicts
git pull origin main

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Function to check if process is running
is_process_running() {
    pgrep -f "python -u main.py" >/dev/null
    return $?
}

# Kill the old server process if running
echo "Stopping the current running process (if any)..."
pkill -f "python -u main.py" || echo "No running process found."

# Wait for process to terminate with timeout
TIMEOUT=30
COUNTER=0
while is_process_running && [ $COUNTER -lt $TIMEOUT ]; do
    echo "Waiting for process to terminate... ($COUNTER/$TIMEOUT)"
    COUNTER=$((COUNTER + 1))
    sleep 1
done

if is_process_running; then
    echo "Failed to terminate process within timeout. Force killing..."
    pkill -9 -f "python -u main.py"
fi

# Clean up GPU resources
echo "Cleaning up GPU resources..."
sudo fuser -k /dev/nvidia* || echo "No running GPU process found."

# Wait for GPU resources to be released
COUNTER=0
while sudo fuser /dev/nvidia* >/dev/null 2>&1 && [ $COUNTER -lt $TIMEOUT ]; do
    echo "Waiting for GPU resources to be released... ($COUNTER/$TIMEOUT)"
    COUNTER=$((COUNTER + 1))
    sleep 1
done

# Start the server with retry logic
echo "Starting the server..."
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i of $MAX_RETRIES to start server..."
    nohup python -u main.py > server.log 2>&1 &
    PID=$!
    
    # Wait briefly to check if process is still running
    sleep 2
    if kill -0 $PID 2>/dev/null; then
        echo "Server started successfully (PID: $PID)"
        # Check if log file contains any immediate errors
        if grep -i "error" server.log >/dev/null 2>&1; then
            echo "Warning: Error detected in server log. Check server.log for details."
        fi
        break
    else
        echo "Attempt $i: Server failed to start."
        if [ $i -eq $MAX_RETRIES ]; then
            echo "Failed to start server after $MAX_RETRIES attempts."
            exit 1
        fi
        echo "Retrying in 5 seconds..."
        sleep 5
    fi
done

echo "Deployment complete. Server log available at: $REPO_DIR/server.log"
echo "To monitor the server, use: tail -f server.log"


nohup python -m uvicorn webhook_listener:app --port=8000 --host 0.0.0.0 >> webhook.log 2>&1 &

echo $! > webhook.pid
echo "Webhook listener started. PID: $(cat webhook.pid)"