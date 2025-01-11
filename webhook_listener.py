from fastapi import FastAPI, HTTPException
import subprocess
import os
import time
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

def get_total_gpu_memory(gpu_index=0):
    try:
        return torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return 0

def get_available_gpu_memory(gpu_index=0):
    try:
        total = torch.cuda.get_device_properties(gpu_index).total_memory
        used = torch.cuda.memory_allocated(gpu_index)
        available = (total - used) / (1024**3)  # Convert to GB
        return available
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return 0

# Function to wait until sufficient GPU memory is available
def wait_for_gpu_memory(min_free_memory_gb, interval_seconds=5, timeout_seconds=10):
    total_memory = get_total_gpu_memory()
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        available_memory = get_available_gpu_memory()

        if available_memory >= min_free_memory_gb:
            return True
        print(f"Waiting for GPU memory... Available: {available_memory:.2f} GB, Required: {min_free_memory_gb} GB")
        time.sleep(interval_seconds)
    return False

@app.post("/webhook")
async def webhook():
    try:
        MIN_FREE_MEMORY_GB = 38.0

        # Wait for sufficient GPU memory
        if wait_for_gpu_memory(MIN_FREE_MEMORY_GB, 5, 10):
            # Pull the latest changes
            subprocess.call(['./deploy.sh'], cwd=REPO_PATH)
            return {"message": "Deployment successful!"}
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Timed out waiting for sufficient GPU memory. Required: {MIN_FREE_MEMORY_GB} GB."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)