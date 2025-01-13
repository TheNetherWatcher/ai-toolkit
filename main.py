import redis
import subprocess
import selectors
import psutil
import json
from datetime import datetime
import time


class TrainingOrchestrator:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='54.90.142.38',
            port=6379,
            decode_responses=True,
            username='default',
            password='ankit'
        )
        self.min_memory_required = 30  # GB
        self.vm_id = self.get_vm_identifier()
        self.training_queue_key = 'training_queue'
        self.log_channel_prefix = 'training_logs'

    def get_vm_identifier(self):
        return f"vm_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(self)}"

    
    def run_training(self, job, training_id):
        """Execute Cog training command with parameters"""
        try:
            log_channel = f"{self.log_channel_prefix}_{training_id}"
            
            # Publish initial log using pub/sub
            self.redis_client.publish(log_channel, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'message': 'Training initialization started',
                'type': 'info'
            }))

            training_params = {
                'dataset_url': job['images_url'],
                'steps': job['steps'],
                'learning_rate': job['learning_rate'],
                'batch_size': job['batch_size'],
                'lora_rank': job['lora_rank'],
                'trigger_word': job['trigger_word'],
                'caption_dropout_rate': job['caption_dropout_rate'],
                'hf_repo_id': job['huggingface_repo_id'],
                "layers_numbers": f"{','.join(map(str, job['specific_layers_trained']))}" or ""
            }
            
            print("------------------")
            print(training_params)

            command = ['python', 'train.py']
            for key, value in training_params.items():
                command.extend([f'--{key}={value}'])

            self.redis_client.hset(f"metadata_{training_id}", "status", 'started')
            
            # Publish command preparation log
            command_str = ' '.join(command)
            print(f"Executing command: {command_str}")
            self.redis_client.publish(log_channel, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'message': f"Executing command: {command_str}",
                'type': 'info'
            }))

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffering for real-time logs
            )

            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            sel.register(process.stderr, selectors.EVENT_READ)

            # Update metadata without logs
            self.redis_client.hset(f"metadata_{training_id}", mapping={
                'started_by': self.vm_id,
                'started_at': datetime.now().isoformat()
            })

            error_occurred = False
            active_streams = 2

            while active_streams > 0:
                if self.should_stop_training(training_id):
                    process.terminate()
                    time.sleep(2) 
                    if process.poll() is None:
                        process.kill()
                    
                    try:
                        subprocess.run('sudo fuser -k /dev/nvidia*', 
                                    shell=True,
                                    check=False,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
                        print("[DEBUG] GPU memory cleared successfully")
                        
                        time.sleep(1)
                        memory_info = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',
                                                   shell=True,
                                                   capture_output=True,
                                                   text=True).stdout.strip()
                        print(f"[DEBUG] Current GPU memory usage: {memory_info}MB")
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to clear GPU memory: {str(e)}")
                    
                    # Update status to stopped
                    self.redis_client.hset(f"metadata_{training_id}", mapping={
                        'status': 'stopped',
                        'stopped_at': datetime.now().isoformat(),
                        'stopped_by': self.vm_id
                    })
                    
                    # Publish stop message
                    self.redis_client.publish(log_channel, json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'message': f"Training {training_id} was stopped manually",
                        'type': 'info'
                    }))
                    return False

                events = sel.select(timeout=1)
                if not events:  # Timeout without events, continue waiting
                    if process.poll() is None:  # Process still running
                        continue
                
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:  # Stream closed
                        sel.unregister(key.fileobj)
                        active_streams -= 1
                        continue
                    
                    timestamp = datetime.now().isoformat()
                    if key.fileobj is process.stdout:
                        log_type = 'info'
                        message = line.strip()
                    elif key.fileobj is process.stderr:
                        if "ERROR" in line.upper() or "FATAL" in line.upper():
                            log_type = 'error'
                            message = f"ERROR: {line.strip()}"
                            error_occurred = True
                        else:
                            log_type = 'warning'
                            message = f"STDERR: {line.strip()}"

                    # Publish log entry using pub/sub
                    self.redis_client.publish(log_channel, json.dumps({
                        'timestamp': timestamp,
                        'message': message,
                        'type': log_type
                    }))
                    print(f"[{timestamp}] {message}", flush=True)

            # Make sure process is actually done
            return_code = process.wait()
            print(f"Process completed with return code: {return_code}")
            
            if error_occurred or return_code != 0:
                raise RuntimeError(f"Training process exited with errors. Return code: {return_code}")

            # Publish completion message
            self.redis_client.publish(log_channel, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'message': f"Training {training_id} completed successfully",
                'type': 'success'
            }))

            # Update final metadata without logs
            self.redis_client.hset(f"metadata_{training_id}", mapping={
                'status': 'completed',
                'completed_by': self.vm_id,
                'completed_at': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            # Publish error message
            self.redis_client.publish(log_channel, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'message': f"Training {training_id} failed: {str(e)}",
                'type': 'error'
            }))

            # Update error metadata without logs
            self.redis_client.hset(f"metadata_{training_id}", mapping={
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            return False
    
    def get_available_memory(self):
        """Returns available memory in GB"""
        return psutil.virtual_memory().available / (1024 ** 3)

    def should_stop_training(self, training_id):
        """Check Redis for stop signal"""
        status = self.redis_client.hget(f"metadata_{training_id}", "status")
        return status == "stopping"

    def start_monitoring(self):
        """Main monitoring loop"""
        print(f"Starting training monitor on VM {self.vm_id}...")
        while True:
            try:
                # Reset variables for new iteration
                job = None
                training_id = None
                
                job = self.get_next_training_job()
                if not job:
                    time.sleep(5)
                    continue

                training_id = job.get('trainingId')
                if not training_id:
                    print("No training ID found in job, skipping")
                    time.sleep(5)
                    continue

                if self.get_available_memory() < self.min_memory_required:
                    self.redis_client.lpush(self.training_queue_key, json.dumps(job))
                    print(f"Insufficient memory for training {training_id}, returning to queue")
                    time.sleep(10)
                    continue

                print(f"Starting training {training_id} with parameters:", job)
                
                training_result = self.run_training(job, training_id)
                print(f"Training {training_id} completed with result: {training_result}")
                
                
                if training_result is not None:  # Training has either succeeded or failed
                    self.cleanup_training_resources(training_id)

            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)
                
                
    def cleanup_training_resources(self, training_id):
        """Clean up any resources after training completion"""
        try:
            # Clear any temporary files or resources
            print(f"[DEBUG] Starting cleanup for training {training_id}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Check and clear any lingering processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.cmdline()
                    if 'cog' in cmdline and str(training_id) in ' '.join(cmdline):
                        print(f"[DEBUG] Killing lingering process: {proc.pid}")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Reset memory
            available_memory = self.get_available_memory()
            print(f"[DEBUG] Available memory after cleanup: {available_memory}GB")
        
        except Exception as e:
            print(f"[ERROR] Cleanup failed for training {training_id}: {str(e)}")

    def get_next_training_job(self):
        """Atomically get and remove the next job from the queue using RPOP"""
        job_data = self.redis_client.rpop(self.training_queue_key)
        if job_data:
            try:
                return json.loads(job_data)
            except json.JSONDecodeError:
                print(f"Error parsing job data: {job_data}")
                return None
        return None

if __name__ == "__main__":
    orchestrator = TrainingOrchestrator()
    orchestrator.start_monitoring()