HOOK_PATH=".git/hooks/post-merge"
if [ ! -f "$HOOK_PATH" ]; then
    echo '#!/bin/bash' > "$HOOK_PATH"
    echo 'chmod +x deploy.sh' >> "$HOOK_PATH"
    chmod +x "$HOOK_PATH"
fi
chmod +x ./deploy.sh

git config pull.rebase false

# Pull the latest changes
git reset --hard
git pull

# Clean up GPU resources
sudo fuser -k /dev/nvidia*

# Install Python dependencies
git submodule update --init --recursive && pip install -r requirements.txt

# Kill any existing webhook processes
pkill -f "uvicorn webhook_listener:app"
sleep 2

# Start webhook listener with proper daemon management
python webhook_automation.py
nohup python -m uvicorn webhook_listener:app --port=8000 --host 0.0.0.0 >> webhook.log 2>&1 &

# Save the PID of the webhook process
echo $! > webhook.pid

echo "Setup completed successfully!"